from __future__ import annotations

"""
Lightweight Groq Chat client with a compatible interface to OllamaClient.
Used when LLM_PROVIDER=groq (or GROQ_API_KEY is set).
"""

import os
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

import requests


class GroqClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_s: int = 60,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is required when using GroqClient")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

        self._stream_lock = threading.Lock()
        self._active_resp: Optional[requests.Response] = None

    # ------------------------------------------------------------
    # Public API (mirrors OllamaClient)
    # ------------------------------------------------------------
    def cancel_active_stream(self) -> None:
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
        keep_alive: Optional[int] = None,  # unused; kept for API parity
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload.update(_map_options(options))

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=_headers(self.api_key),
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Groq response missing choices: {data}")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected Groq message shape: {data}")
        return content

    def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[int] = None,  # unused; kept for parity
        stop_event: Optional[threading.Event] = None,
        read_timeout_s: Optional[float] = None,
        max_stall_s: float = 30.0,
    ) -> Iterator[str]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload.update(_map_options(options))

        connect_timeout = min(10.0, float(self.timeout_s))
        timeout = (connect_timeout, read_timeout_s)

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=_headers(self.api_key),
            stream=True,
            timeout=timeout,
        )
        resp.raise_for_status()

        with self._stream_lock:
            self._active_resp = resp

        last_activity = [time.time()]

        def _watchdog():
            while True:
                if stop_event is not None and stop_event.is_set():
                    return
                with self._stream_lock:
                    active = self._active_resp
                if active is None:
                    return
                if time.time() - last_activity[0] > max_stall_s:
                    try:
                        active.close()
                    except Exception:
                        pass
                    return
                time.sleep(0.5)

        threading.Thread(target=_watchdog, daemon=True).start()

        try:
            for raw in resp.iter_lines(decode_unicode=True):
                if stop_event is not None and stop_event.is_set():
                    break
                if not raw:
                    continue
                if raw.strip() == "data: [DONE]":
                    break
                if not raw.startswith("data:"):
                    continue
                last_activity[0] = time.time()
                try:
                    chunk = raw[len("data:") :].strip()
                    data = requests.utils.json.loads(chunk)
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    token = delta.get("content")
                    if isinstance(token, str) and token:
                        yield token
                except Exception:
                    continue
        finally:
            self.cancel_active_stream()


def _headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _map_options(opts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "temperature" in opts:
        out["temperature"] = opts.get("temperature")
    if "num_predict" in opts:
        out["max_tokens"] = opts.get("num_predict")
    return out
