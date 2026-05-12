from __future__ import annotations

import re
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

def _safe_input_samplerate(device_index: int | None, preferred: int) -> int:
    """
    Pick a samplerate that the device accepts.
    Tries preferred first, then device default, then common rates.
    """
    # Common rates to try (order matters)
    candidates = [preferred, 48000, 44100, 32000, 16000]

    try:
        if device_index is not None:
            d = sd.query_devices(device_index)
        else:
            d = sd.query_devices(sd.default.device[0])  # default input device
        default_sr = int(d.get("default_samplerate") or 0)
        if default_sr and default_sr not in candidates:
            candidates.insert(1, default_sr)
    except Exception:
        pass

    # Try opening a tiny stream to validate
    for sr in candidates:
        try:
            with sd.InputStream(device=device_index, samplerate=sr, channels=1, dtype="float32"):
                return int(sr)
        except Exception:
            continue

    # If nothing works, just return preferred and let the real open raise a clear error
    return int(preferred)


def _resample_linear(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Lightweight resample (no scipy). Good enough for speech.
    x: 1D float32 in [-1, 1]
    """
    orig_sr = int(orig_sr)
    target_sr = int(target_sr)
    if x.size == 0 or orig_sr == target_sr:
        return x.astype(np.float32, copy=False)

    duration = x.shape[0] / float(orig_sr)
    n_out = int(round(duration * target_sr))
    if n_out <= 1:
        return np.array([], dtype=np.float32)

    xp = np.linspace(0.0, duration, num=x.shape[0], endpoint=False)
    fp = x.astype(np.float32, copy=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)

    y = np.interp(x_new, xp, fp).astype(np.float32)
    return y


CLINIC_CANONICAL = {
    # common near-misses -> correct
    r"\bdr\.?\s*awad\b": "Dr. Amal",
    r"\bdr\.?\s*ahmad\b": "Dr. Amal",
    r"\bdoctor\s+awad\b": "Doctor Amal",
    r"\bdoctor\s+ahmad\b": "Doctor Amal",
}

# phrases that often come out wrong in your logs
PHRASE_FIXES = [
    (r"\bincubate is\b", "the date is"),
    (r"\bi warned him that\b", "i want"),
    (r"\bcan't want to\b", "i want to"),
]


class WhisperSTT:
    def __init__(
        self,
        model_size: str = "medium.en",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        input_device: int | None = None,
        target_sample_rate: int = 16000,

    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.input_device = input_device
        self.target_sample_rate = int(target_sample_rate)

        self.model = WhisperModel(
            model_size_or_path=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def _record_until_silence(
        self,
        max_seconds: float = 8.0,
        silence_seconds: float = 1.15,
        start_threshold: float = 0.0035,
        stop_threshold: float = 0.0020,
        chunk_ms: int = 30,
        pre_roll_ms: int = 450,
        min_speech_ms: int = 550,
    ) -> np.ndarray:
        """
        Working baseline:
        - not too aggressive (doesn't cut early)
        - not too long (reduces hallucinations)
        """
        target_sr = self.target_sample_rate
        stream_sr = _safe_input_samplerate(self.input_device, preferred=target_sr)
        sr = stream_sr  # (used for recording chunk math)

        chunk = int(sr * (chunk_ms / 1000.0))
        max_chunks = int(max_seconds * sr / chunk)
        silence_needed = int(silence_seconds * sr / chunk)
        pre_roll = int(pre_roll_ms * sr / chunk)

        frames: list[np.ndarray] = []
        pre: list[np.ndarray] = []
        speaking = False
        silence_count = 0

        speech_frames = 0
        min_speech_needed = int((min_speech_ms / 1000.0) * sr / chunk)


        with sd.InputStream(
            device=self.input_device,
            samplerate=sr,
            channels=1,
            dtype="float32",
        ) as stream:
            for _ in range(max_chunks):
                data, _ = stream.read(chunk)
                x = data[:, 0]

                rms = float(np.sqrt(np.mean(x * x) + 1e-12))

                # Adaptive pre-gain for quiet mics (e.g., some earbud/HFP mics).
                # This only affects the VAD decision + stored waveform; final audio
                # is still normalized later.
                gain = 1.0
                if not speaking and rms > 1e-6:
                    # If we're close to the start threshold, gently boost for detection.
                    if rms < start_threshold and rms >= (start_threshold * 0.35):
                        gain = min(3.0, start_threshold / (rms + 1e-9))
                if gain != 1.0:
                    x = x * gain
                    rms = float(np.sqrt(np.mean(x * x) + 1e-12))


                pre.append(x.copy())
                if len(pre) > pre_roll:
                    pre.pop(0)

                if not speaking:
                    if rms >= start_threshold:
                        speaking = True
                        frames.extend(pre)
                        frames.append(x)
                        silence_count = 0
                else:
                    frames.append(x)
                    speech_frames += 1

                    # Don't allow early stop until we have enough speech
                    if speech_frames < min_speech_needed:
                        silence_count = 0
                    else:
                        if rms < stop_threshold:
                            silence_count += 1
                        else:
                            silence_count = 0

                        if silence_count >= silence_needed:
                            break


        if not frames:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(frames).astype(np.float32)

        # Energy guard: avoid treating very low-energy captures (often TTS tail/noise) as speech
        peak0 = float(np.max(np.abs(audio)) + 1e-9)
        rms0 = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        if rms0 < 0.01 and peak0 < 0.25:
            return np.array([], dtype=np.float32)

        # Peak normalize
        peak = float(np.max(np.abs(audio)) + 1e-9)
        if peak > 0:
            audio = audio / peak

        # RMS normalize / AGC
        # Only apply strong gain when we already have real speech energy;
        # otherwise we can amplify leakage/noise into a fake 'heard' transcript.
        rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        target_rms = 0.08
        if rms >= 0.015:
            gain = min(target_rms / rms, 6.0)
            audio = audio * gain

        audio = np.clip(audio, -1.0, 1.0)

        peak2 = float(np.max(np.abs(audio)) + 1e-9)
        rms2 = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        print(f"🎚️ audio len={audio.shape[0]/target_sr:.2f}s peak={peak2:.3f} rms={rms2:.3f}")


        # If device records at 48k/44.1k, convert to 16k for Whisper
        if stream_sr != target_sr:
            audio = _resample_linear(audio, stream_sr, target_sr)
            print(f"🔁 Resampled {stream_sr} -> {target_sr} Hz")


        return audio

    def _postprocess(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""

        # phrase fixes (light, deterministic)
        low = t.lower()
        for pat, rep in PHRASE_FIXES:
            low = re.sub(pat, rep, low, flags=re.IGNORECASE)
        t = low

        # canonical doctor fixes
        for pat, rep in CLINIC_CANONICAL.items():
            t = re.sub(pat, rep, t, flags=re.IGNORECASE)

        # phone join
        digits_all = re.sub(r"\D+", "", t)
        if len(digits_all) >= 10:
            # keep last 10 digits (fixes extra leading junk like 1/8)
            last10 = digits_all[-10:]
            t = re.sub(r"\d{10,}", last10, t)

        # normalize time phrases
        t = re.sub(r"\b([1-9])\s*(a\.?m\.?|am)\b", r"0\1:00", t, flags=re.IGNORECASE)
        t = re.sub(r"\b(10|11|12)\s*(a\.?m\.?|am)\b", r"\1:00", t, flags=re.IGNORECASE)

        # 9 30 am -> 09:30
        t = re.sub(
            r"\b(0?\d|1[0-2])\s*(?:\:)?\s*(\d{2})\s*(a\.?m\.?|am)\b",
            lambda m: f"{int(m.group(1)):02d}:{m.group(2)}",
            t,
            flags=re.IGNORECASE,
        )

        # 11 o'clock -> 11:00
        t = re.sub(r"\b(\d{1,2})\s*o'?clock\b", r"\1:00", t, flags=re.IGNORECASE)

        return t.strip()

    def listen_once(self, seconds: float = 6.0, silence_seconds: float = 1.15) -> str:
        max_seconds = float(seconds)
        silence_seconds = float(silence_seconds)

        print(f"🎤 Speak now... (GPU STT, max {max_seconds:.1f}s)")
        audio = self._record_until_silence(
            max_seconds=max_seconds,
            silence_seconds=silence_seconds,
        )

        # Pad audio so Whisper doesn't clip first/last words
        pad_ms = 180
        pad = np.zeros(int(self.target_sample_rate * pad_ms / 1000.0), dtype=np.float32)
        audio = np.concatenate([pad, audio, pad])


        if audio.size == 0:
            print("🗣️  Heard: ")
            return ""

        # Keep these conservative. Your GPU is fast enough.
        segments, _info = self.model.transcribe(
            audio,
            language="en",
            beam_size=6,
            best_of=6,
            temperature=0.0,
            vad_filter=True,  # you already do silence stop
            condition_on_previous_text=False,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        text = self._postprocess(text)

        print(f"🗣️  Heard: {text}")
        return text

    def listen_continuous(
        self,
        on_utterance,
        stop_event,
        *,
        max_seconds: float = 8.0,
        silence_seconds: float = 1.15,
        start_threshold: float = 0.0035,
        stop_threshold: float = 0.0020,
        chunk_ms: int = 30,
        pre_roll_ms: int = 450,
        min_speech_ms: int = 550,
        allow_listen=None,
    ) -> None:
        """
        Continuously listen on an open mic stream and call on_utterance(audio_f32, sr)
        when a full utterance ends. Uses the same thresholds as _record_until_silence.
        """
        target_sr = self.target_sample_rate
        stream_sr = _safe_input_samplerate(self.input_device, preferred=target_sr)
        sr = stream_sr

        chunk = int(sr * (chunk_ms / 1000.0))
        max_chunks = int(max_seconds * sr / chunk)
        silence_needed = int(silence_seconds * sr / chunk)
        pre_roll = int(pre_roll_ms * sr / chunk)

        frames: list[np.ndarray] = []
        pre: list[np.ndarray] = []
        speaking = False
        silence_count = 0
        speech_frames = 0
        min_speech_needed = int((min_speech_ms / 1000.0) * sr / chunk)
        utt_chunks = 0

        with sd.InputStream(
            device=self.input_device,
            samplerate=sr,
            channels=1,
            dtype="float32",
        ) as stream:
            while not stop_event.is_set():
                if allow_listen is not None and not allow_listen():
                    # Drop any partial utterance while paused
                    frames.clear()
                    pre.clear()
                    speaking = False
                    silence_count = 0
                    speech_frames = 0
                    utt_chunks = 0
                    try:
                        stream.read(chunk)
                    except Exception:
                        pass
                    continue

                try:
                    data, _ = stream.read(chunk)
                except Exception:
                    continue

                x = data[:, 0]
                rms = float(np.sqrt(np.mean(x * x) + 1e-12))

                # Adaptive pre-gain for quiet mics (same as _record_until_silence)
                gain = 1.0
                if not speaking and rms > 1e-6:
                    if rms < start_threshold and rms >= (start_threshold * 0.35):
                        gain = min(3.0, start_threshold / (rms + 1e-9))
                if gain != 1.0:
                    x = x * gain
                    rms = float(np.sqrt(np.mean(x * x) + 1e-12))

                pre.append(x.copy())
                if len(pre) > pre_roll:
                    pre.pop(0)

                if not speaking:
                    if rms >= start_threshold:
                        speaking = True
                        frames.extend(pre)
                        frames.append(x)
                        silence_count = 0
                        speech_frames = 0
                        utt_chunks = 0
                else:
                    frames.append(x)
                    speech_frames += 1
                    utt_chunks += 1

                    if speech_frames < min_speech_needed:
                        silence_count = 0
                    else:
                        if rms < stop_threshold:
                            silence_count += 1
                        else:
                            silence_count = 0

                    if silence_count >= silence_needed or utt_chunks >= max_chunks:
                        if frames:
                            audio = np.concatenate(frames).astype(np.float32)

                            peak0 = float(np.max(np.abs(audio)) + 1e-9)
                            rms0 = float(np.sqrt(np.mean(audio * audio) + 1e-12))
                            if rms0 >= 0.01 or peak0 >= 0.25:
                                peak = float(np.max(np.abs(audio)) + 1e-9)
                                if peak > 0:
                                    audio = audio / peak

                                rms2 = float(np.sqrt(np.mean(audio * audio) + 1e-12))
                                target_rms = 0.08
                                if rms2 >= 0.015:
                                    gain2 = min(target_rms / rms2, 6.0)
                                    audio = audio * gain2

                                audio = np.clip(audio, -1.0, 1.0)

                                peak2 = float(np.max(np.abs(audio)) + 1e-9)
                                rms3 = float(np.sqrt(np.mean(audio * audio) + 1e-12))
                                print(f"??? audio len={audio.shape[0]/target_sr:.2f}s peak={peak2:.3f} rms={rms3:.3f}")

                                if stream_sr != target_sr:
                                    audio = _resample_linear(audio, stream_sr, target_sr)
                                    print(f"?? Resampled {stream_sr} -> {target_sr} Hz")

                                on_utterance(audio, target_sr)

                        frames.clear()
                        speaking = False
                        silence_count = 0
                        speech_frames = 0
                        utt_chunks = 0

    def transcribe_audio(self, audio_f32: "np.ndarray", sample_rate: int) -> str:
        """Transcribe an already-captured float32 mono signal.

        This is used for barge-in: we capture the user's interruption audio
        (including a short pre-roll) and feed it directly to Whisper. That avoids
        missing the first words ("no tell me...").
        """
        import numpy as np

        if audio_f32 is None:
            return ""

        audio = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return ""

        # Resample to Whisper target SR if needed.
        if int(sample_rate) != int(self.target_sample_rate):
            audio = _resample_linear(audio, int(sample_rate), int(self.target_sample_rate))
            pad_ms = 180
            pad = np.zeros(int(self.target_sample_rate * pad_ms / 1000.0), dtype=np.float32)
            audio = np.concatenate([pad, audio, pad])


        segments, _info = self.model.transcribe(
            audio,
            language="en",
            beam_size=6,
            best_of=6,
            temperature=0.0,
            vad_filter=True,    
            condition_on_previous_text=False,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        return self._postprocess(text)


def _resample_linear(x: "np.ndarray", sr_in: int, sr_out: int) -> "np.ndarray":
    """Simple linear resampler (no extra deps). Good enough for short barge audio."""
    import numpy as np

    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32, copy=False)

    ratio = float(sr_out) / float(sr_in)
    n_out = max(1, int(round(x.size * ratio)))
    t_in = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    y = np.interp(t_out, t_in, x).astype(np.float32)
    return y
