"""
LiveKit ↔ local brain connector.

Responsibilities:
- Receive user transcripts from LiveKit.
- Run them through the existing voice brain (Ollama + router/tooling).
- Chunk assistant replies for snappy playback.
- Synthesize audio with Piper and stream it back to LiveKit.
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import threading
import tempfile
import time
import wave
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, List, Optional

import numpy as np

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    room_io,
)
from livekit.agents import inference, stt as lk_stt
from livekit.agents.stt import StreamAdapter
from livekit.agents.utils import merge_frames
from livekit.plugins import silero
from faster_whisper import WhisperModel
from livekit.agents.types import NOT_GIVEN, DEFAULT_API_CONNECT_OPTIONS

try:
    # When imported as part of the `src` package (e.g. `python -m src.voice_server`)
    from .agent_core import run_agent_once
    from .logging.chat_logger import ChatLogConfig, log_turn, new_session_id
    from .ollama_client import OllamaClient
    from .groq_client import GroqClient
    from .router.session import SessionState
except ImportError:  # pragma: no cover
    # When imported as a top-level module (e.g. `sys.path` includes `src/`)
    from agent_core import run_agent_once
    from logging.chat_logger import ChatLogConfig, log_turn, new_session_id
    from ollama_client import OllamaClient
    from groq_client import GroqClient
    from router.session import SessionState


ROOT = Path(__file__).resolve().parents[1]

# Default HF cache under the repo’s models/hub folder.
DEFAULT_HF_CACHE = ROOT / "models" / "hub"
DEFAULT_HF_CACHE.mkdir(parents=True, exist_ok=True)
if not os.getenv("HUGGINGFACE_HUB_CACHE"):
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(DEFAULT_HF_CACHE)
    try:
        print(f"[CONFIG] HUGGINGFACE_HUB_CACHE set to {os.environ['HUGGINGFACE_HUB_CACHE']}", flush=True)
    except Exception:
        pass

# Small helper to surface the chosen STT config in console output.
def _log_whisper_config(model: str, device: str, compute_type: str, where: str) -> None:
    try:
        print(f"[CONFIG] whisper backend={where} model={model} device={device} compute_type={compute_type}", flush=True)
    except Exception:
        pass


def _log_whisper_loaded(model: WhisperModel, where: str) -> None:
    try:
        print(f"[CONFIG] whisper loaded={model.model_size} backend={where}", flush=True)
    except Exception:
        pass


def _lang_from_env() -> Optional[str]:
    lang = os.getenv("WHISPER_LANGUAGE")
    if lang is None:
        return None
    lang = lang.strip()
    return lang or None


def _highpass_filter(signal: np.ndarray, cutoff_hz: float, fs: int) -> np.ndarray:
    """First-order high-pass; cutoff_hz <= 0 disables."""
    if cutoff_hz <= 0 or fs <= 0 or signal.size == 0:
        return signal
    rc = 1.0 / (2.0 * np.pi * float(cutoff_hz))
    dt = 1.0 / float(fs)
    alpha = rc / (rc + dt)
    out = np.empty_like(signal, dtype=np.float32)
    out[0] = signal[0]
    for n in range(1, signal.size):
        out[n] = alpha * (out[n - 1] + signal[n] - signal[n - 1])
    return out


def _rms_normalize(signal: np.ndarray, target_dbfs: Optional[float]) -> np.ndarray:
    """RMS normalize to target_dbfs; None disables; boost limited to 4x to avoid noise lift."""
    if target_dbfs is None or signal.size == 0:
        return signal
    rms = np.sqrt(np.mean(np.square(signal), dtype=np.float64))
    if rms <= 1e-9:
        return signal
    current_db = 20.0 * np.log10(rms + 1e-12)
    gain_db = target_dbfs - current_db
    gain_lin = 10.0 ** (gain_db / 20.0)
    gain_lin = min(gain_lin, 4.0)
    out = signal * gain_lin
    np.clip(out, -1.0, 1.0, out=out)
    return out


def split_into_tts_chunks(text: str) -> List[str]:
    """Deterministic chunking: sentence-ish pieces for TTS playback."""
    s = (text or "").strip()
    if not s:
        return []

    parts = re.split(r"(?<=[.!?])\s+", s)
    chunks = [p.strip() for p in parts if p and p.strip()]

    # shorter chunks so audio starts streaming sooner
    if len(chunks) == 1 and len(chunks[0]) > 160:
        long = chunks[0]
        chunks = [long[i : i + 140].strip() for i in range(0, len(long), 140) if long[i : i + 140].strip()]

    return chunks


def _lowpass_filter(signal: np.ndarray, cutoff_hz: float, fs: int) -> np.ndarray:
    """First-order low-pass; cutoff_hz <= 0 disables."""
    if cutoff_hz <= 0 or fs <= 0 or signal.size == 0:
        return signal
    rc = 1.0 / (2.0 * np.pi * float(cutoff_hz))
    dt = 1.0 / float(fs)
    alpha = dt / (rc + dt)
    out = np.empty_like(signal, dtype=np.float32)
    out[0] = signal[0]
    for n in range(1, signal.size):
        out[n] = out[n - 1] + alpha * (signal[n] - out[n - 1])
    return out


def _speech_boost(signal: np.ndarray, fs: int, f_low: float, f_high: float, gain_db: float) -> np.ndarray:
    """Simple band-pass boost for presence; disabled if gain_db == 0."""
    if gain_db == 0 or signal.size == 0 or fs <= 0 or f_low <= 0 or f_high <= f_low:
        return signal
    nyq = fs / 2.0
    low = max(min(f_low / nyq, 0.99), 0.0)
    high = max(min(f_high / nyq, 0.99), low + 1e-4)
    # Biquad bandpass (second-order); simple IIR via scipy-like coefficients
    omega = np.pi * (low + high)
    bw = np.pi * (high - low)
    alpha = np.sin(omega) * np.sinh(np.log(2) / 2 * bw / np.sin(omega))
    cosw = np.cos(omega)
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1 + alpha
    a1 = -2 * cosw
    a2 = 1 - alpha
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0
    out = np.empty_like(signal, dtype=np.float32)
    x1 = x2 = y1 = y2 = 0.0
    for i, x0 in enumerate(signal):
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[i] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
    gain_lin = 10.0 ** (gain_db / 20.0)
    return (signal + (out * (gain_lin - 1.0))).astype(np.float32)


def _initial_prompt_from_env() -> str:
    """
    Optional initial prompt to bias Whisper toward key names/terms.

    Env vars:
      WHISPER_INITIAL_PROMPT   free-form text (e.g., "Transcribe clearly.")
      WHISPER_HOTWORDS        comma/semicolon list of names/keywords to keep exact
    """
    prompt = os.getenv("WHISPER_INITIAL_PROMPT", "").strip()
    hotwords = os.getenv("WHISPER_HOTWORDS", "").strip()

    words: List[str] = []
    if hotwords:
        words = [w.strip() for w in hotwords.replace(";", ",").split(",") if w.strip()]

    parts: List[str] = []
    if prompt:
        parts.append(prompt)
    if words:
        names = ", ".join(words)
        parts.append(f"Key names/keywords to keep exact: {names}.")

    return " ".join(parts).strip()

def _env_or_default(name: str, default: str) -> str:
    val = os.getenv(name)
    if val is None or val == "":
        try:
            print(f"[CONFIG] {name} not set; using default '{default}'", flush=True)
        except Exception:
            pass
        return default
    return val


# ---------------------------------------------------------------------------
# Brain: wraps the existing LLM/tool/router logic in a single callable.
# ---------------------------------------------------------------------------
class Brain:
    def __init__(
        self,
        model: str = "llama3.1:8b",
        keep_alive: int = -1,
        history_max: int = 10,
        warmup: bool = True,
        mode: str = "chat",
    ) -> None:
        self.client = OllamaClient()
        self.model = model
        self.keep_alive = keep_alive
        self.session = SessionState(max_history=history_max)
        self.base_prompt = (ROOT / "configs" / "agents" / "base.txt").read_text(encoding="utf-8")
        self.role_prompt = (ROOT / "configs" / "agents" / "role.txt").read_text(encoding="utf-8")
        self.mode = (mode or "chat").strip().lower()
        self.messages: List[dict] = []
        self._msg_lock = threading.Lock()

        data_dir = ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        self.session_id = new_session_id()
        (data_dir / "last_session.txt").write_text(self.session_id, encoding="utf-8")
        self.log_cfg = ChatLogConfig(csv_path=data_dir / "chat_log.csv")

        # Warm the model once so first user turn is fast.
        if warmup:
            try:
                self.client.chat(
                    self.model,
                    [{"role": "system", "content": "warmup"}],
                    keep_alive=self.keep_alive,
                )
            except Exception:
                pass

    def _trim_messages(self) -> None:
        try:
            max_hist = int(getattr(self.session, "max_history", 0) or 0)
        except Exception:
            max_hist = 0
        if max_hist > 0 and len(self.messages) > max_hist:
            self.messages = self.messages[-max_hist:]

    def _chat_system_prompt(self) -> str:
        # Fast path: role-only. Avoid base.txt JSON/tool formatting overhead.
        return (self.role_prompt or "").strip()

    def _chat_options(self) -> dict:
        def _get_int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except Exception:
                return int(default)

        def _get_float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except Exception:
                return float(default)

        return {
            "temperature": _get_float("CHAT_TEMPERATURE", 0.2),
            "num_predict": _get_int("CHAT_NUM_PREDICT", 80),
            "num_ctx": _get_int("CHAT_NUM_CTX", 1024),
        }

    def start_chat_stream(self, user_text: str):
        clean = (user_text or "").strip()
        if not clean:
            return iter(())

        with self._msg_lock:
            self.messages.append({"role": "user", "content": clean})
            self._trim_messages()
            log_turn(
                self.log_cfg,
                session_id=self.session_id,
                role="user",
                text=clean,
                meta={"client": "livekit", "stt": "livekit"},
            )

        sys_prompt = self._chat_system_prompt()
        with self._msg_lock:
            system_msgs = ([{"role": "system", "content": sys_prompt}] if sys_prompt else [])
            send_messages = system_msgs + list(self.messages)
        return self.client.chat_stream(
            self.model,
            send_messages,
            options=self._chat_options(),
            keep_alive=self.keep_alive,
        )

    def finish_chat_stream(self, assistant_text: str) -> None:
        answer = (assistant_text or "").strip()
        if not answer:
            return
        with self._msg_lock:
            self.messages.append({"role": "assistant", "content": answer})
            self._trim_messages()
            log_turn(
                self.log_cfg,
                session_id=self.session_id,
                role="assistant",
                text=answer,
                meta={"client": "livekit", "model": self.model, "mode": "chat"},
            )

    def append_assistant_partial(self, assistant_text: str) -> None:
        """Persist assistant text that was already spoken before barge-in."""
        clean = (assistant_text or "").strip()
        if not clean:
            return
        with self._msg_lock:
            last = self.messages[-1] if self.messages else None
            if isinstance(last, dict) and last.get("role") == "assistant":
                last_txt = (last.get("content") or "").strip()
                if last_txt == clean:
                    return

            self.messages.append({"role": "assistant", "content": clean})
            self._trim_messages()
            log_turn(
                self.log_cfg,
                session_id=self.session_id,
                role="assistant",
                text=clean,
                meta={"client": "livekit", "model": self.model, "mode": "chat", "partial": True},
            )

    def pop_last_user_if_matches(self, user_text: str) -> bool:
        """Remove the most recent user message if it matches (used when user barges before any assistant output)."""
        clean = (user_text or "").strip()
        if not clean:
            return False
        with self._msg_lock:
            if not self.messages:
                return False
            last = self.messages[-1]
            if not isinstance(last, dict):
                return False
            if last.get("role") != "user":
                return False
            if (last.get("content") or "").strip() != clean:
                return False
            self.messages.pop()
            return True


    def __call__(self, text: str) -> Optional[str]:
        clean = (text or "").strip()
        if not clean:
            return None

        self.messages.append({"role": "user", "content": clean})
        log_turn(
            self.log_cfg,
            session_id=self.session_id,
            role="user",
            text=clean,
            meta={"client": "livekit", "stt": "livekit"},
        )

        answer = run_agent_once(
            client=self.client,
            base_prompt=self.base_prompt,
            role_prompt=self.role_prompt,
            messages=self.messages,
            session=self.session,
            model=self.model,
            use_router=True,
            use_base=True,
        )

        answer = (answer or "").strip()
        if not answer:
            return None

        self.messages.append({"role": "assistant", "content": answer})
        log_turn(
            self.log_cfg,
            session_id=self.session_id,
            role="assistant",
            text=answer,
            meta={"client": "livekit", "model": self.model},
        )
        return answer


# ---------------------------------------------------------------------------
# Piper synthesis -> PCM frames for LiveKit playback.
# ---------------------------------------------------------------------------
class PiperSynth:
    def __init__(
        self,
        exe: Path,
        model: Path,
        config: Optional[Path] = None,
        noise_scale: float = 0.6,
        length_scale: float = 1.05,
        use_cuda: bool = False,
        threads: Optional[int] = None,
    ) -> None:
        self.exe = Path(exe)
        self.model = Path(model)
        self.config = Path(config) if config else None
        self.noise_scale = noise_scale
        self.length_scale = length_scale
        self.use_cuda = use_cuda
        self.threads = threads

    def synthesize_wav(self, text: str) -> bytes:
        """Return a WAV byte stream for the given text."""
        txt = (text or "").strip()
        if not txt:
            return b""

        fd, wav_path = tempfile.mkstemp(prefix="piper_", suffix=".wav")
        os.close(fd)
        cmd = [
            str(self.exe),
            "--model",
            str(self.model),
            "--output_file",
            wav_path,
            "--noise_scale",
            str(self.noise_scale),
            "--length_scale",
            str(self.length_scale),
        ]
        if self.use_cuda:
            cmd.append("--cuda")
        if self.config:
            cmd += ["--config", str(self.config)]
        if self.threads and self.threads > 0:
            cmd += ["--threads", str(int(self.threads))]

        import subprocess

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )

        try:
            assert proc.stdin is not None
            proc.stdin.write(txt + "\n")
            proc.stdin.flush()
            proc.stdin.close()
            proc.wait(timeout=30)
        finally:
            try:
                proc.kill()
            except Exception:
                pass

        try:
            data = Path(wav_path).read_bytes()
        finally:
            try:
                Path(wav_path).unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
        return data


def _wav_bytes_to_frames(data: bytes, frame_ms: int = 20) -> List[rtc.AudioFrame]:
    if not data:
        return []
    with wave.open(io.BytesIO(data), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16)
    elif sampwidth == 4:
        samples = (np.frombuffer(raw, dtype=np.int32) >> 16).astype(np.int16)
    else:
        samples = np.frombuffer(raw, dtype=np.int8).astype(np.int16)

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)

    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    frames: List[rtc.AudioFrame] = []
    for i in range(0, len(samples), frame_len):
        chunk = samples[i : i + frame_len]
        if not chunk.size:
            continue
        frames.append(
            rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=chunk.shape[0],
            )
        )
    return frames


async def _frames_stream(frames: List[rtc.AudioFrame]) -> AsyncIterator[rtc.AudioFrame]:
    # Hand frames over promptly. LiveKit's audio source buffers them and plays
    # them out at exact realtime, applying backpressure when its queue is full.
    # Manual per-frame asyncio.sleep() pacing is unreliable on Windows (~15 ms
    # timer granularity) and caused playout-buffer underruns -> choppy speech.
    for frame in frames:
        yield frame
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# LiveKit wiring.
# ---------------------------------------------------------------------------
def _build_brain() -> Brain:
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    keep_alive = int(os.getenv("OLLAMA_KEEP_ALIVE", "-1"))
    history_max = int(os.getenv("HISTORY_MAX", "10"))
    warmup = os.getenv("OLLAMA_WARMUP", "1").lower() not in {"0", "false", "no"}
    mode = os.getenv("BRAIN_MODE", "chat").strip().lower()
    if mode not in {"chat", "router"}:
        mode = "chat"
    brain = Brain(model=model, keep_alive=keep_alive, history_max=history_max, warmup=warmup, mode=mode)

    # Warm streaming path once so first token latency is lower after cold start.
    def _warm_stream() -> None:
        try:
            for _ in brain.client.chat_stream(
                brain.model,
                [{"role": "user", "content": "warm"}],
                options={"num_predict": 8},
                keep_alive=brain.keep_alive,
                read_timeout_s=5,
            ):
                break
        except Exception:
            pass

    threading.Thread(target=_warm_stream, daemon=True).start()
    return brain


def _build_tts() -> PiperSynth:
    exe = Path(os.getenv("PIPER_EXE", ROOT / "bin" / "piper.exe"))
    model = Path(os.getenv("PIPER_MODEL", ROOT / "models" / "piper" / "en_IN_navguru.onnx"))
    config = Path(os.getenv("PIPER_CONFIG", ROOT / "models" / "piper" / "en_IN_navguru.onnx.json"))
    def _f(env_name: str, default: float) -> float:
        try:
            return float(os.getenv(env_name, str(default)))
        except Exception:
            return default

    use_cuda = os.getenv("PIPER_USE_CUDA", "0").lower() in {"1", "true", "yes"}
    try:
        threads = int(os.getenv("PIPER_THREADS", "0"))
    except Exception:
        threads = 0
    noise_scale = _f("PIPER_NOISE_SCALE", 0.6)
    length_scale = _f("PIPER_LENGTH_SCALE", 1.05)

    synth = PiperSynth(
        exe=exe,
        model=model,
        config=config,
        noise_scale=noise_scale,
        length_scale=length_scale,
        use_cuda=use_cuda,
        threads=threads if threads > 0 else None,
    )

    # Warm up once to remove the first-call stall.
    try:
        _ = synth.synthesize_wav("warmup")
    except Exception:
        pass

    return synth


class LocalWhisperSTT(lk_stt.STT):
    """Non-streaming Whisper wrapped by StreamAdapter for VAD-based turn detection."""

    def __init__(
        self,
        model: WhisperModel,
        *,
        target_sr: int = 16000,
        language: Optional[str] = None,
    ) -> None:
        super().__init__(
            capabilities=lk_stt.STTCapabilities(
                streaming=False, interim_results=False, diarization=False, offline_recognize=True
            )
        )
        self._model = model
        self._target_sr = int(target_sr)
        self._language = language
        self._ap = None
        cleaner = (os.getenv("CLEANER") or "").strip().lower()
        if cleaner == "webrtc":
            try:
                import webrtc_audio_processing as ap

                self._ap = ap.AudioProcessing(
                    enable_ns=True,
                    enable_agc=False,
                    enable_hpf=True,
                    enable_vad=False,
                    noise_suppression_level=getattr(ap, "NS_LEVEL_HIGH", 2),
                )
                try:
                    print("[CONFIG] cleaner=webrtc_ns", flush=True)
                except Exception:
                    pass
            except Exception:
                self._ap = None

    @property
    def model(self) -> str:  # type: ignore[override]
        try:
            return str(self._model.model_size)
        except Exception:
            return "whisper"

    @property
    def provider(self) -> str:  # type: ignore[override]
        return "faster-whisper"

    async def _recognize_impl(
        self,
        buffer,
        *,
        language=NOT_GIVEN,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
    ) -> lk_stt.SpeechEvent:
        stt_t0 = time.monotonic()
        # Merge frames and resample to target_sr.
        merged = merge_frames(buffer)
        input_sr = merged.sample_rate
        resampler = None
        if input_sr != self._target_sr:
            resampler = rtc.AudioResampler(
                input_rate=input_sr,
                output_rate=self._target_sr,
                quality=rtc.AudioResamplerQuality.MEDIUM,
            )

        samples: list[np.ndarray] = []
        frames = [merged] if isinstance(merged, rtc.AudioFrame) else [merged]
        for fr in frames:
            if resampler:
                for r in resampler.push(fr):
                    samples.append(np.frombuffer(r.data, dtype=np.int16))
            else:
                samples.append(np.frombuffer(fr.data, dtype=np.int16))
        if resampler:
            for r in resampler.flush():
                samples.append(np.frombuffer(r.data, dtype=np.int16))

        if not samples:
            return lk_stt.SpeechEvent(
                type=lk_stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[lk_stt.SpeechData(language="", text="")],
        )

        audio_i16 = np.concatenate(samples).astype(np.int16)
        # Optional denoise using WebRTC AudioProcessing (noise suppression + HPF).
        if self._ap is not None:
            try:
                frame_len = 160  # 10 ms at 16 kHz
                cleaned: list[np.ndarray] = []
                for i in range(0, len(audio_i16), frame_len):
                    frame = audio_i16[i : i + frame_len]
                    if frame.size < frame_len:
                        frame = np.pad(frame, (0, frame_len - frame.size), mode="constant")
                    buf = frame.astype(np.int16)[np.newaxis, :]
                    out = self._ap.process_stream(buf)
                    cleaned.append(out[0, : frame.size])
                audio_i16 = np.concatenate(cleaned).astype(np.int16)
            except Exception:
                pass

        audio_f32 = (audio_i16.astype(np.float32)) / 32768.0

        # Optional pre-roll pad to avoid clipping initial phonemes.
        try:
            pre_roll_ms = int(os.getenv("WHISPER_PRE_ROLL_MS", "180"))
        except Exception:
            pre_roll_ms = 180
        if pre_roll_ms > 0:
            pad = np.zeros(int(self._target_sr * pre_roll_ms / 1000.0), dtype=np.float32)
            audio_f32 = np.concatenate([pad, audio_f32])

        # Input conditioning: optional HPF and RMS leveling to tame rumble and volume swings.
        try:
            hpf_cutoff = float(os.getenv("WHISPER_HPF_HZ", "120") or 0)
        except Exception:
            hpf_cutoff = 120.0
        try:
            lpf_cutoff = float(os.getenv("WHISPER_LPF_HZ", "8000") or 0)
        except Exception:
            lpf_cutoff = 8000.0
        try:
            boost_db = float(os.getenv("WHISPER_BOOST_DB", "3.0") or 0.0)
            boost_lo = float(os.getenv("WHISPER_BOOST_LOW_HZ", "2000") or 2000.0)
            boost_hi = float(os.getenv("WHISPER_BOOST_HIGH_HZ", "4000") or 4000.0)
        except Exception:
            boost_db, boost_lo, boost_hi = 3.0, 2000.0, 4000.0
        try:
            tgt_dbfs_env = os.getenv("WHISPER_TARGET_RMS_DB", "")
            target_dbfs = float(tgt_dbfs_env) if tgt_dbfs_env.strip() != "" else None
        except Exception:
            target_dbfs = None
        if hpf_cutoff > 0:
            audio_f32 = _highpass_filter(audio_f32, cutoff_hz=hpf_cutoff, fs=self._target_sr)
        if lpf_cutoff > 0:
            audio_f32 = _lowpass_filter(audio_f32, cutoff_hz=lpf_cutoff, fs=self._target_sr)
        if boost_db != 0.0:
            audio_f32 = _speech_boost(audio_f32, fs=self._target_sr, f_low=boost_lo, f_high=boost_hi, gain_db=boost_db)
        if target_dbfs is not None:
            audio_f32 = _rms_normalize(audio_f32, target_dbfs=target_dbfs)

        lang = None if language is NOT_GIVEN else language
        if not lang:
            lang = self._language
        if not lang:
            lang = None

        # Tunables from env: beam size & temperature.
        try:
            beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "4"))
        except Exception:
            beam_size = 4
        try:
            temperature = float(os.getenv("WHISPER_TEMPERATURE", "0.0"))
        except Exception:
            temperature = 0.0
        initial_prompt = _initial_prompt_from_env() or None

        segments, _info = self._model.transcribe(
            audio_f32,
            language=lang,
            beam_size=beam_size,
            temperature=temperature,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
        )
        stt_ms = (time.monotonic() - stt_t0) * 1000.0
        try:
            print(f"[STT_METRIC] stt_ms={stt_ms:.0f}")
        except Exception:
            pass
        text_parts = [seg.text.strip() for seg in segments if seg.text]
        full_text = " ".join(text_parts).strip()

        return lk_stt.SpeechEvent(
            type=lk_stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                lk_stt.SpeechData(
                    language=lang or "",
                    text=full_text,
                    confidence=0.0,
                )
            ],
        )


def _silero_opts_from_env() -> dict:
    """Read Silero VAD tuning knobs from env (seconds)."""
    def _f(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default

    # Convert ms helpers
    def _ms(name: str, default_ms: float) -> float:
        return _f(name, default_ms) / 1000.0

    act = _f("SILERO_ACTIVATION", 0.5)
    deact = _f("SILERO_DEACTIVATION", max(act - 0.15, 0.01))
    return {
        "min_speech_duration": _ms("SILERO_MIN_SPEECH_MS", 50.0),
        "min_silence_duration": _ms("SILERO_MIN_SILENCE_MS", 550.0),
        "prefix_padding_duration": _ms("SILERO_PREFIX_PAD_MS", 500.0),
        "activation_threshold": act,
        "deactivation_threshold": deact,
    }


def _build_stt(ctx: JobContext) -> lk_stt.STT:
    backend = os.getenv("STT_BACKEND", "whisper").lower()
    if backend == "livekit":
        model = os.getenv("LIVEKIT_STT_MODEL", "deepgram/nova-3-general")
        return inference.STT(model=model)

    # Whisper path
    w_model = ctx.proc.userdata.get("whisper_model")
    if w_model is None:
        default_model_path = str(DEFAULT_HF_CACHE / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo")
        w_model_name = _env_or_default("WHISPER_MODEL", default_model_path)
        device = _env_or_default("WHISPER_DEVICE", "cpu")
        compute_type = _env_or_default(
            "WHISPER_COMPUTE_TYPE",
            "int8" if device == "cpu" else "int8_float16",
        )
        _log_whisper_config(w_model_name, device, compute_type, where="session build")
        w_model = WhisperModel(
            model_size_or_path=w_model_name,
            device=device,
            compute_type=compute_type,
        )
        _log_whisper_loaded(w_model, where="session build")
    silero_dev = os.getenv("SILERO_DEVICE", "cpu").strip().lower()
    force_cpu = silero_dev != "cuda"
    vad = ctx.proc.userdata.get("silero_vad")
    if vad is None:
        vad = silero.VAD.load(
            sample_rate=16000,
            force_cpu=force_cpu,
            **_silero_opts_from_env(),
        )
    lang = _lang_from_env()
    base_stt = LocalWhisperSTT(model=w_model, target_sr=16000, language=lang)
    return StreamAdapter(stt=base_stt, vad=vad)


# Allow slower startup when loading large Whisper models (default is 10s).
server = AgentServer(initialize_process_timeout=300.0)


def _prewarm(proc: JobProcess) -> None:
    # Load heavy models once per worker process to avoid first-turn lag.
    backend = os.getenv("STT_BACKEND", "whisper").lower()
    if backend == "whisper":
        default_model_path = str(DEFAULT_HF_CACHE / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo")
        w_model = _env_or_default("WHISPER_MODEL", default_model_path)
        device = _env_or_default("WHISPER_DEVICE", "cpu")
        compute_type = _env_or_default("WHISPER_COMPUTE_TYPE", "int8" if device == "cpu" else "int8_float16")
        _log_whisper_config(w_model, device, compute_type, where="prewarm")
        proc.userdata["whisper_model"] = WhisperModel(
            model_size_or_path=w_model,
            device=device,
            compute_type=compute_type,
        )
        _log_whisper_loaded(proc.userdata["whisper_model"], where="prewarm")
        silero_dev = os.getenv("SILERO_DEVICE", "cpu").strip().lower()
        force_cpu = silero_dev != "cuda"
        proc.userdata["silero_vad"] = silero.VAD.load(
            sample_rate=16000,
            force_cpu=force_cpu,
            **_silero_opts_from_env(),
        )


server.setup_fnc = _prewarm


@server.rtc_session()
async def voice_session(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    # A dedicated VAD for the session lets it detect the user starting to speak
    # and cut off TTS within ~one word, instead of waiting for the full Whisper
    # transcript (+650 ms of trailing silence + decode). The StreamAdapter keeps
    # its own separate VAD for STT segmentation.
    bargein_vad = ctx.proc.userdata.get("silero_vad")
    try:
        barge_min_sec = float(os.getenv("BARGE_MIN_SEC", "0.2"))
    except Exception:
        barge_min_sec = 0.2

    session = AgentSession(
        stt=_build_stt(ctx),
        vad=bargein_vad,
        # How long the user must be speaking before TTS is cut off. Low = snappy
        # barge-in; raise it if background noise / "um" stops the bot too easily.
        min_interruption_duration=barge_min_sec,
        # We manage interruptions explicitly in this file; auto-resume can create
        # confusing "skip" behavior after interruptions.
        resume_false_interruption=False,
    )

    brain = _build_brain()
    tts = _build_tts()
    current_turn: Optional[asyncio.Task] = None
    turn_seq = 0
    active_turn_id = 0

    @dataclass
    class TurnState:
        turn_id: int
        user_text: str
        last_user_ts: float
        assistant_text: str = ""
        assistant_spoken: str = ""

    active_state: Optional[TurnState] = None

    async def _send_chat(role: str, text: str) -> None:
        """Send a chat message over LiveKit data to populate frontend transcript."""
        try:
            msg = {
                "id": f"{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}",
                "timestamp": int(time.time() * 1000),
                "message": text or "",
                "type": "chatMessage",
                # optional role marker for debugging; frontend ignores
                "role": role,
            }
            payload = json.dumps(msg).encode("utf-8")
            # Legacy chat topic used by LiveKit components
            await ctx.room.local_participant.publish_data(payload, reliable=True, topic="lk-chat-topic")
            # New chat topic used by newer components
            await ctx.room.local_participant.publish_data(payload, reliable=True, topic="lk.chat")
        except Exception:
            pass

    def _log(tag: str, text: str) -> None:
        clean = " ".join((text or "").split())
        if not clean:
            return
        print(f"[{tag}] {clean}")

    try:
        fragment_merge_sec = float(os.getenv("USER_FRAGMENT_MERGE_SEC", "1.2"))
    except Exception:
        fragment_merge_sec = 1.2

    preemptive_stt = str(os.getenv("PREEMPTIVE_STT", "1")).lower() not in {"0", "false", "no"}

    def _set_current_turn(task: asyncio.Task) -> None:
        nonlocal current_turn

        def _clear(_task: asyncio.Task) -> None:
            nonlocal current_turn
            if current_turn is _task:
                if _task.cancelled():
                    _log("TURN", "cancelled")
                elif _task.exception():
                    _log("TURN_ERR", repr(_task.exception()))
                current_turn = None

        current_turn = task
        task.add_done_callback(_clear)

    def _tts_chunks(text: str, *, max_chars: int = 240) -> List[str]:
        # Keep chunks as large as possible (whole sentences) so playback has the
        # fewest seams. Only hard-wrap a sentence that is longer than max_chars,
        # and wrap it at word boundaries rather than mid-word.
        chunks = split_into_tts_chunks(text)
        adjusted: List[str] = []
        for c in chunks:
            c = (c or "").strip()
            if not c:
                continue
            if len(c) <= max_chars:
                adjusted.append(c)
                continue
            words = c.split()
            cur = ""
            for w in words:
                if cur and len(cur) + 1 + len(w) > max_chars:
                    adjusted.append(cur)
                    cur = w
                else:
                    cur = (cur + " " + w).strip()
            if cur:
                adjusted.append(cur)
        return adjusted

    def _strip_markdown(text: str) -> str:
        """Remove basic markdown markers so TTS doesn't speak symbols."""
        if not text:
            return ""
        clean = re.sub(r"[*`_]+", "", text)  # bold/italic/code markers
        # Remove leading list markers like "1.", "1)", "-", "*" (but do not break decimals like "3.14").
        clean = re.sub(r"^[\s]*(?:[-*•]|[0-9]{1,3}[.)])(?:\s+|$)", "", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    async def _say_chunk(chunk: str) -> None:
        wav_bytes = await asyncio.to_thread(tts.synthesize_wav, chunk)
        frames = await asyncio.to_thread(_wav_bytes_to_frames, wav_bytes)
        if not frames:
            print("[VOICE_SERVER] warn: empty audio frames")
            return
        handle = session.say(
            text=chunk,
            audio=_frames_stream(frames),
            allow_interruptions=True,  # allow barge-in by interrupting current speech
            add_to_chat_ctx=True,  # surface in frontend chat log
        )
        # Wait until this chunk is fully played out (or interrupted).
        await handle

    async def _run_turn(state: TurnState) -> None:
        cancel_event = asyncio.Event()
        text = state.user_text
        turn_t0 = time.monotonic()
        llm_t0: Optional[float] = None
        llm_t1: Optional[float] = None
        tts_t0: Optional[float] = None
        tts_t1: Optional[float] = None
        first_tok_ms: Optional[float] = None

        def _is_active() -> bool:
            return (not cancel_event.is_set()) and (state.turn_id == active_turn_id)

        async def _speaker_worker(sentence_q: "asyncio.Queue[Optional[str]]") -> None:
            nonlocal tts_t0, tts_t1

            # Flatten incoming sentences into a queue of TTS-sized pieces so the
            # synthesizer can run one piece ahead of playback.
            piece_q: "asyncio.Queue[Optional[str]]" = asyncio.Queue()

            async def _feeder() -> None:
                try:
                    while True:
                        sent = await sentence_q.get()
                        if sent is None:
                            break
                        speak_text = _strip_markdown(sent)
                        if not speak_text:
                            continue
                        for piece in _tts_chunks(speak_text):
                            await piece_q.put(piece)
                finally:
                    await piece_q.put(None)

            async def _synth(piece: str) -> List[rtc.AudioFrame]:
                wav = await asyncio.to_thread(tts.synthesize_wav, piece)
                return await asyncio.to_thread(_wav_bytes_to_frames, wav)

            feeder_task = asyncio.create_task(_feeder())
            pending: "Optional[asyncio.Task]" = None
            pending_text = ""
            try:
                first = await piece_q.get()
                if first is not None and _is_active():
                    pending = asyncio.create_task(_synth(first))
                    pending_text = first

                while pending is not None:
                    try:
                        frames = await pending
                    except asyncio.CancelledError:
                        return
                    cur_text = pending_text

                    # Begin playback immediately; it streams out at realtime in
                    # the background while we synthesize the next piece, so there
                    # is no silent gap between pieces.
                    play_handle = None
                    if frames and _is_active():
                        if tts_t0 is None:
                            tts_t0 = time.monotonic()
                        state.assistant_spoken = (state.assistant_spoken + " " + cur_text).strip()
                        play_handle = session.say(
                            text=cur_text,
                            audio=_frames_stream(frames),
                            allow_interruptions=True,
                            add_to_chat_ctx=True,
                        )

                    nxt = await piece_q.get()
                    if nxt is not None and _is_active():
                        pending = asyncio.create_task(_synth(nxt))
                        pending_text = nxt
                    else:
                        pending = None

                    if play_handle is not None:
                        await play_handle
                        tts_t1 = time.monotonic()

                    if not _is_active():
                        if pending is not None:
                            pending.cancel()
                            pending = None
                        return
            except asyncio.CancelledError:
                if pending is not None:
                    pending.cancel()
                return
            finally:
                feeder_task.cancel()

        try:
            if brain.mode == "chat":
                loop = asyncio.get_running_loop()
                token_q: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
                sentence_q: "asyncio.Queue[Optional[str]]" = asyncio.Queue()

                token_iter = brain.start_chat_stream(text)
                llm_t0 = time.monotonic()

                def _llm_worker() -> None:
                    try:
                        for tok in token_iter:
                            if not _is_active():
                                break
                            loop.call_soon_threadsafe(token_q.put_nowait, tok)
                    finally:
                        loop.call_soon_threadsafe(token_q.put_nowait, None)

                threading.Thread(target=_llm_worker, daemon=True).start()
                speaker_task = asyncio.create_task(_speaker_worker(sentence_q))

                answer_parts: List[str] = []
                buf = ""
                spoke_first = False
                while True:
                    tok = await token_q.get()
                    if tok is None or not _is_active():
                        break
                    if first_tok_ms is None and llm_t0 is not None:
                        first_tok_ms = (time.monotonic() - llm_t0) * 1000
                        try:
                            first_from_stt = (time.monotonic() - state.last_user_ts) * 1000
                            _log(
                                "LLM_FIRST_TOKEN",
                                f"turn_id={state.turn_id} first_ms={first_tok_ms:.0f} first_from_stt_ms={first_from_stt:.0f}",
                            )
                        except Exception:
                            pass
                    answer_parts.append(tok)
                    state.assistant_text += tok
                    buf += tok
                    # Flush completed sentences to the speaker as soon as they form.
                    while True:
                        m = re.search(r"(.+?[.!?])\s+", buf)
                        if not m:
                            break
                        sent = m.group(1).strip()
                        buf = buf[m.end():]
                        if sent:
                            await sentence_q.put(sent)
                            spoke_first = True
                    # Nothing spoken yet but we already have a clause's worth of
                    # text: speak the opening clause now so audio starts sooner.
                    if not spoke_first and len(buf) >= 50:
                        head = buf[:80]
                        m = re.search(r"^(.{15,}?[,;:\-—])\s", head)
                        cut = m.end(1) if m else None
                        if cut is None:
                            sp = head.rfind(" ")
                            cut = sp if sp >= 30 else None
                        if cut:
                            sent = buf[:cut].strip()
                            buf = buf[cut:].lstrip()
                            if sent:
                                await sentence_q.put(sent)
                                spoke_first = True

                tail = buf.strip()
                if tail and not cancel_event.is_set():
                    await sentence_q.put(tail)

                await sentence_q.put(None)
                await speaker_task
                llm_t1 = time.monotonic()

                if not _is_active():
                    return

                answer = "".join(answer_parts).strip()
                if _is_active():
                    brain.finish_chat_stream(answer)
                    if answer:
                        _log("MODEL", answer)
                        asyncio.create_task(_send_chat("assistant", answer))

                # metrics
                try:
                    total_ms = (time.monotonic() - turn_t0) * 1000
                    llm_ms = (llm_t1 - llm_t0) * 1000 if llm_t0 and llm_t1 else -1
                    tts_ms = (tts_t1 - tts_t0) * 1000 if tts_t0 and tts_t1 else -1
                    _log("TURN_METRIC", f"turn_id={state.turn_id} llm_ms={llm_ms:.0f} tts_ms={tts_ms:.0f} total_ms={total_ms:.0f}")
                except Exception:
                    pass
                return

            # Router/tools mode (slower, but supports tool workflows).
            llm_t0 = time.monotonic()
            reply = await asyncio.to_thread(brain, text)
            llm_t1 = time.monotonic()
            if not reply or not _is_active():
                return

            if _is_active():
                _log("MODEL", reply)
                asyncio.create_task(_send_chat("assistant", reply))
            chunks = _tts_chunks(reply)
            for chunk in chunks:
                if not _is_active():
                    break
                await _say_chunk(chunk)
                if tts_t0 is None:
                    tts_t0 = time.monotonic()
                tts_t1 = time.monotonic()
            try:
                total_ms = (time.monotonic() - turn_t0) * 1000
                llm_ms = (llm_t1 - llm_t0) * 1000 if llm_t0 and llm_t1 else -1
                tts_ms = (tts_t1 - tts_t0) * 1000 if tts_t0 and tts_t1 else -1
                _log("TURN_METRIC", f"turn_id={state.turn_id} llm_ms={llm_ms:.0f} tts_ms={tts_ms:.0f} total_ms={total_ms:.0f}")
            except Exception:
                pass
        except asyncio.CancelledError:
            cancel_event.set()
            # unblock queues quickly
            try:
                token_q.put_nowait(None)  # type: ignore[name-defined]
            except Exception:
                pass
            try:
                sentence_q.put_nowait(None)  # type: ignore[name-defined]
            except Exception:
                pass
            raise

    async def _handle_transcript(evt) -> None:
        nonlocal current_turn, turn_seq, active_turn_id, active_state
        # evt has: text, is_final
        is_final = bool(getattr(evt, "is_final", False))

        text = getattr(evt, "text", None) or getattr(evt, "transcript", None)
        if not text:
            return
        text = str(text).strip()
        if not text:
            return

        now = time.monotonic()
        prev_state = active_state
        is_interrupt = bool(current_turn and not current_turn.done())
        # "Barge" is *only* when the agent is actively speaking audio.
        is_speaking = False
        try:
            is_speaking = session.agent_state == "speaking"
        except Exception:
            is_speaking = False
        is_barge = is_interrupt and is_speaking

        # Kick off LLM early on partials if nothing is running.
        if not is_final and preemptive_stt:
            if not current_turn or current_turn.done():
                _log("USER_PARTIAL", text)
                turn_seq += 1
                active_turn_id = turn_seq
                active_state = TurnState(turn_id=active_turn_id, user_text=text, last_user_ts=now)
                new_task = asyncio.create_task(_run_turn(active_state))
                _set_current_turn(new_task)
            else:
                if active_state:
                    active_state.user_text = text
            return

        # Each new transcript starts a new turn id; older turns should self-stop.
        turn_seq += 1
        active_turn_id = turn_seq

        # Cancel and interrupt any ongoing turn/speech/LLM stream.
        if is_interrupt:
            # Stop speech immediately if the agent was speaking.
            try:
                speech = session.current_speech
                if speech is not None and hasattr(speech, "interrupt") and not speech.done():
                    speech.interrupt(force=True)
            except Exception:
                pass
            try:
                brain.client.cancel_active_stream()
            except Exception:
                pass

            _log("BARGE" if is_barge else "USER", text)
            if is_barge and prev_state is not None:
                spoken_dbg = (prev_state.assistant_spoken or "").strip()
                if spoken_dbg:
                    _log("MODEL_BEFORE_BARGE", spoken_dbg)

            if prev_state is not None:
                spoken = (prev_state.assistant_spoken or "").strip()
                if spoken:
                    brain.append_assistant_partial(spoken)
                else:
                    # Merge quick follow-up fragments like "No. Can we talk about" + "Mars?"
                    if fragment_merge_sec and (now - float(prev_state.last_user_ts)) <= float(fragment_merge_sec):
                        text = f"{prev_state.user_text.rstrip()} {text}".strip()
                    brain.pop_last_user_if_matches(prev_state.user_text)

            if current_turn and not current_turn.done():
                current_turn.cancel()
        else:
            _log("USER", text)
            asyncio.create_task(_send_chat("user", text))

        active_state = TurnState(turn_id=active_turn_id, user_text=text, last_user_ts=now)
        new_task = asyncio.create_task(_run_turn(active_state))
        _set_current_turn(new_task)

    @session.on("user_input_transcribed")
    def _on_transcript(evt) -> None:
        # Event emitter expects sync callbacks; dispatch async work as a task.
        asyncio.create_task(_handle_transcript(evt))

    agent = Agent(instructions="You are a transport-only agent. Core reasoning is handled externally.")
    room_opts = room_io.RoomOptions(close_on_disconnect=False)
    await session.start(agent=agent, room=ctx.room, room_options=room_opts)
    await ctx.connect()


def main() -> None:
    cli.run_app(server)


if __name__ == "__main__":
    main()
