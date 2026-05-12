# src/piper_tts.py
from __future__ import annotations

import os
import queue
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None


@dataclass
class PiperConfig:
    piper_exe: str                   # path to piper.exe
    model_path: str                  # .onnx
    config_path: Optional[str] = None  # .onnx.json (optional; piper often infers, but keep it explicit)
    sample_rate: Optional[int] = None  # if None, read from wav
    volume: float = 1.0              # 0.0 .. 1.0+
    sentence_silence: float = 0.0    # extra silence between sentences (seconds)


class PiperTTS:
    """
    Piper-based TTS with the SAME public API as your KokoroTTS:

      speak_chunks(chunks) -> speak_id
      wait_started(speak_id)
      wait_done(speak_id)
      stop()
      is_speaking()
      close()

    Implementation:
    - Worker thread runs piper.exe to synthesize a temp wav.
    - Plays wav via sounddevice (stop() interrupts playback quickly).
    """

    def __init__(
        self,
        piper_exe,
        model_path,
        config_path,
        volume=1.0,
        noise_scale=0.6,
        length_scale=1.05,
        sentence_silence=0.05,
    ):
        if sd is None:
            raise RuntimeError("sounddevice is required for PiperTTS. Install: pip install sounddevice")

        self.cfg = PiperConfig(
            piper_exe=os.path.abspath(piper_exe),
            model_path=os.path.abspath(model_path),
            config_path=os.path.abspath(config_path) if config_path else None,
            volume=float(volume),
            sentence_silence=float(sentence_silence),
        )

        self.noise_scale = noise_scale
        self.length_scale = length_scale

        self._cmd_q: "queue.Queue[Tuple[Any, ...]]" = queue.Queue()
        self._evt_q: "queue.Queue[Tuple[Any, ...]]" = queue.Queue()

        self._next_id = 1
        self._speaking = False

        self._last_started: dict[int, int] = {}
        self._done: set[int] = set()
        self._stopped: set[int] = set()

        self._quit = threading.Event()
        self._stop_now = threading.Event()



        # Current subprocess (so stop() can terminate generation too)
        self._active_proc_lock = threading.Lock()
        self._active_proc: Optional[subprocess.Popen] = None

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ---------------- Public API ----------------

    def speak_chunks(self, chunks: List[str]) -> int:
        speak_id = self._next_id
        self._next_id += 1

        self._last_started.pop(speak_id, None)
        self._done.discard(speak_id)
        self._stopped.discard(speak_id)

        text = " ".join((c or "").strip() for c in (chunks or []) if (c or "").strip())
        self._cmd_q.put(("speak", speak_id, text))
        return speak_id

    def wait_started(self, speak_id: int, timeout: float = 1.0) -> bool:
        t0 = time.time()
        while (time.time() - t0) < timeout:
            self._drain()
            if speak_id in self._last_started:
                return True
            time.sleep(0.01)
        return False

    def wait_done(self, speak_id: int, timeout: float = 30.0) -> str:
        """
        Returns: "done" | "stopped" | "timeout"
        """
        t0 = time.time()
        while (time.time() - t0) < timeout:
            self._drain()
            if speak_id in self._done:
                return "done"
            if speak_id in self._stopped:
                return "stopped"
            time.sleep(0.02)
        return "timeout"

    def stop(self) -> None:
        # stop playback + cancel current generation
        self._cmd_q.put(("stop",))

    def is_speaking(self) -> bool:
        self._drain()
        return self._speaking

    def close(self) -> None:
        self._quit.set()
        self._cmd_q.put(("quit",))
        try:
            self._thread.join(timeout=1.5)
        except Exception:
            pass

    # ---------------- Internals ----------------

    def _emit(self, evt: Tuple[Any, ...]) -> None:
        try:
            self._evt_q.put_nowait(evt)
        except Exception:
            pass

    def _set_speaking(self, val: bool) -> None:
        self._speaking = bool(val)
        self._emit(("speaking", self._speaking, time.time()))

    def _drain(self) -> None:
        while True:
            try:
                evt = self._evt_q.get_nowait()
            except Exception:
                break

            kind = evt[0]
            if kind == "speaking":
                self._speaking = bool(evt[1])
            elif kind == "started":
                self._last_started[int(evt[1])] = 0
            elif kind == "done":
                self._done.add(int(evt[1]))
            elif kind == "stopped":
                self._stopped.add(int(evt[1]))
            elif kind == "error":
                # You can print(evt) here if you want debugging
                pass

    def _terminate_active_proc(self) -> None:
        with self._active_proc_lock:
            p = self._active_proc
            self._active_proc = None
        if p is None:
            return
        try:
            p.terminate()
        except Exception:
            pass

    def _read_wav_float32(self, wav_path: str) -> Tuple[np.ndarray, int]:
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            nchan = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)

        if sampwidth == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            # fallback: treat as unsigned bytes
            audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128.0) / 128.0

        if nchan > 1:
            audio = audio.reshape(-1, nchan).mean(axis=1)

        # volume
        audio *= float(self.cfg.volume)

        # avoid clipping
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio /= peak

        return audio, sr

    def _play_audio_blocking(self, audio: np.ndarray, sr: int) -> None:
        if audio.size == 0:
            return

        self._stop_now.clear()
        sd.stop()  # stop anything already playing

        sd.play(audio, sr, blocking=False)

        # Wait in small steps so stop() can interrupt quickly
        while sd.get_stream() is not None and sd.get_stream().active:
            if self._quit.is_set() or self._stop_now.is_set():
                sd.stop()
                break
            time.sleep(0.01)

    def _worker(self) -> None:
        active_id: Optional[int] = None

        while not self._quit.is_set():
            try:
                cmd = self._cmd_q.get(timeout=0.1)
            except Exception:
                continue

            if not cmd:
                continue

            op = cmd[0]

            if op == "quit":
                break

            if op == "stop":
                self._stop_now.set()
                self._terminate_active_proc()
                try:
                    sd.stop()
                except Exception:
                    pass

                if active_id is not None:
                    self._emit(("stopped", active_id, time.time()))
                active_id = None
                self._set_speaking(False)
                continue

            if op == "speak":
                speak_id = int(cmd[1])
                text = (cmd[2] or "").strip()
                active_id = speak_id

                if not text:
                    self._emit(("done", speak_id, time.time()))
                    active_id = None
                    self._set_speaking(False)
                    continue

                # Create a temp wav output path
                fd, wav_path = tempfile.mkstemp(prefix="piper_", suffix=".wav")
                os.close(fd)

                try:
                    self._set_speaking(True)
                    self._emit(("started", speak_id, 0, time.time()))

                    # Build Piper command
                    # Build Piper command
                    p_cmd = [
                        self.cfg.piper_exe,
                        "--model", self.cfg.model_path,
                        "--output_file", wav_path,
                        "--noise_scale", str(self.noise_scale),
                        "--length_scale", str(self.length_scale),
                    ]
                    if self.cfg.config_path:
                        p_cmd += ["--config", self.cfg.config_path]
                    if self.cfg.sentence_silence is not None:
                        p_cmd += ["--sentence_silence", str(self.cfg.sentence_silence)]

                    # Run piper with stdin text (Windows-safe pattern) :contentReference[oaicite:4]{index=4}
                    self._stop_now.clear()
                    proc = subprocess.Popen(
                        p_cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        encoding="utf-8",
                    )
                    with self._active_proc_lock:
                        self._active_proc = proc

                    try:
                        proc.stdin.write(text)
                        proc.stdin.write("\n")
                        proc.stdin.flush()
                        proc.stdin.close()
                    except Exception:
                        pass

                    # Wait for synthesis, allow stop/quit
                    while proc.poll() is None:
                        if self._quit.is_set() or self._stop_now.is_set():
                            self._terminate_active_proc()
                            break
                        time.sleep(0.02)

                    if self._stop_now.is_set() or self._quit.is_set():
                        self._emit(("stopped", speak_id, time.time()))
                        continue

                    # Play wav
                    audio, sr = self._read_wav_float32(wav_path)
                    self._play_audio_blocking(audio, sr)

                    if self._stop_now.is_set() or self._quit.is_set():
                        self._emit(("stopped", speak_id, time.time()))
                    else:
                        self._emit(("done", speak_id, time.time()))

                except Exception as e:
                    self._emit(("error", speak_id, repr(e), time.time()))
                    self._emit(("stopped", speak_id, time.time()))
                finally:
                    with self._active_proc_lock:
                        self._active_proc = None
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass

                    active_id = None
                    self._set_speaking(False)

        # Cleanup
        try:
            sd.stop()
        except Exception:
            pass
