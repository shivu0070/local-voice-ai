from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple
import json
import threading
import time
import os

import requests


class OllamaClient:
    """Minimal Ollama chat client with optional streaming.

    Uses:
      POST {base_url}/api/chat

    Notes:
    - keep_alive must be an integer (e.g. -1) for "keep model warm" behavior.
    - In streaming mode, we keep a reference to the active response so another
      thread (e.g. barge-in) can cancel generation by closing the socket.
    """

    def __init__(self, base_url: Optional[str] = None, timeout_s: int = 120) -> None:
        # Default to 127.0.0.1 (NOT "localhost"): on Windows, Python's requests
        # tries IPv6 ::1 first for "localhost" and stalls ~2.3s before falling
        # back to IPv4 — a fixed per-request tax. Override with OLLAMA_HOST
        # (e.g. "ollama:11434" in Docker).
        if base_url is None:
            base_url = os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434"
        if not base_url.startswith(("http://", "https://")):
            base_url = "http://" + base_url
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

        self._stream_lock = threading.Lock()
        self._active_resp: Optional[requests.Response] = None
        # Keep models warm; default from env or 900s.
        try:
            self.default_keep_alive = int(os.getenv("OLLAMA_KEEP_ALIVE", "900"))
        except Exception:
            self.default_keep_alive = 900

    def cancel_active_stream(self) -> None:
        """Best-effort: immediately stop any in-flight streaming response."""
        with self._stream_lock:
            resp = self._active_resp
            self._active_resp = None

        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[int] = None,
    ) -> str:
        url = f"{self.base_url}/api/chat"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.default_keep_alive if keep_alive is None else keep_alive,
        }
        if options:
            payload["options"] = options

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout_s)
                if resp.status_code >= 500:
                    raise requests.exceptions.HTTPError(f"{resp.status_code} {resp.text}")
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt >= max_retries:
                    raise
                sleep = min(2 ** (attempt - 1), 5)
                print(f"[OLLAMA] chat retry {attempt}/{max_retries} after error: {e}")
                time.sleep(sleep)

        data = resp.json()
        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected Ollama response shape: {data}")
        return content

    def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[int] = None,
        stop_event: Optional[threading.Event] = None,
        read_timeout_s: Optional[float] = None,
        max_stall_s: float = 30.0,
    ) -> Iterator[str]:
        """Stream tokens from Ollama as they arrive.

        Important: `stop_event` alone is not enough to cancel immediately because
        `resp.iter_lines()` can block on socket reads. For truly instant cancel,
        call `cancel_active_stream()` from another thread (barge-in).

        Yields: token strings.
        """

        url = f"{self.base_url}/api/chat"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.default_keep_alive if keep_alive is None else keep_alive,
        }
        if options:
            payload["options"] = options

        _probe_t0 = time.monotonic()
        _probe_first = True

        # Requests timeouts:
        # - connect timeout avoids hanging if Ollama isn't reachable
        # - read timeout is optional; for streaming we typically leave it as None
        #   and rely on cancel_active_stream() to break blocking reads instantly.
        connect_timeout = min(10.0, float(self.timeout_s))
        timeout = (connect_timeout, None if read_timeout_s is None else float(read_timeout_s))

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(url, json=payload, stream=True, timeout=timeout)
                if resp.status_code >= 500:
                    raise requests.exceptions.HTTPError(f"{resp.status_code} {resp.text}")
                resp.raise_for_status()
                break
            except requests.exceptions.ReadTimeout:
                if stop_event is not None and stop_event.is_set():
                    return
                if attempt >= max_retries:
                    raise
            except requests.exceptions.RequestException as e:
                if attempt >= max_retries:
                    raise
            time.sleep(min(2 ** (attempt - 1), 5))

        # Mark active response so another thread can cancel immediately.
        with self._stream_lock:
            self._active_resp = resp

        # Watchdog: if the stream stalls (no bytes/tokens) for too long,
        # cancel it so the main loop never needs Ctrl+C.
        last_activity = [time.time()]

        def _watchdog():
            # Daemon thread.
            while True:
                if stop_event is not None and stop_event.is_set():
                    return
                # If stream already cleared, we're done.
                with self._stream_lock:
                    active = self._active_resp
                if active is None:
                    return
                if max_stall_s and (time.time() - last_activity[0]) > float(max_stall_s):
                    try:
                        if stop_event is not None:
                            stop_event.set()
                    except Exception:
                        pass
                    try:
                        self.cancel_active_stream()
                    except Exception:
                        pass
                    return
                time.sleep(0.2)

        threading.Thread(target=_watchdog, daemon=True).start()

        try:
            # JSONL: each line is a JSON object
            for line in resp.iter_lines(decode_unicode=True):
                if stop_event is not None and stop_event.is_set():
                    break

                last_activity[0] = time.time()
                if not line:
                    continue

                # Ollama streams jsonl: {"message":{"role":"assistant","content":"..."}, "done":false}
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                if obj.get("done"):
                    try:
                        _ms = lambda k: obj.get(k, 0) / 1e6
                        print(
                            f"[OLLAMA_DONE] load_ms={_ms('load_duration'):.0f} "
                            f"prompt_eval_ms={_ms('prompt_eval_duration'):.0f} "
                            f"prompt_tok={obj.get('prompt_eval_count', 0)} "
                            f"eval_ms={_ms('eval_duration'):.0f} "
                            f"eval_tok={obj.get('eval_count', 0)} "
                            f"total_ms={_ms('total_duration'):.0f}",
                            flush=True,
                        )
                    except Exception:
                        pass
                    break

                msg = obj.get("message") or {}
                tok = msg.get("content") or ""
                if tok:
                    if _probe_first:
                        _probe_first = False
                        try:
                            print(f"[OLLAMA_PROBE] post_to_first_token_ms={(time.monotonic()-_probe_t0)*1000:.0f}", flush=True)
                        except Exception:
                            pass
                    yield tok
        except KeyboardInterrupt:
            # Graceful stop on Ctrl+C
            if stop_event is not None:
                stop_event.set()
            return
        except Exception as e:
            # If another thread cancelled the stream by closing the response,
            # urllib3/requests can throw mid-iteration. Treat that as a normal
            # cancellation, not a crash.
            if stop_event is not None and stop_event.is_set():
                return

            with self._stream_lock:
                active = self._active_resp

            if active is None:
                return

            # Common "response closed mid-stream" shapes
            try:
                import urllib3  # type: ignore
                proto_err = getattr(urllib3.exceptions, "ProtocolError", None)
            except Exception:
                proto_err = None

            import requests as _requests  # local alias

            if isinstance(e, AttributeError):
                return
            if isinstance(
                e,
                (
                    _requests.exceptions.ChunkedEncodingError,
                    _requests.exceptions.ConnectionError,
                    _requests.exceptions.StreamConsumedError,
                    OSError,
                    TimeoutError,
                ),
            ):
                return
            if proto_err is not None and isinstance(e, proto_err):
                return

            raise
        finally:
            # Ensure the HTTP connection is closed so generation stops.
            try:
                resp.close()
            except Exception:
                pass

            with self._stream_lock:
                if self._active_resp is resp:
                    self._active_resp = None
