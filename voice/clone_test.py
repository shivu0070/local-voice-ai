"""
XTTS v2 (Coqui) voice-clone test harness — standalone, isolated from the agent.

Clones a voice from a short reference clip, synthesizes text, and reports load +
synthesis latency. Use it to judge JARVIS voice quality and speed BEFORE wiring
XTTS into the live agent. See voice/README.md for setup.

    .\voice\.venv-xtts\Scripts\python voice\clone_test.py \
        --ref voice\refs\jarvis.wav --text "Good evening, sir."
"""
from __future__ import annotations

import argparse
import os
import time
import wave
from pathlib import Path

# Auto-accept the XTTS model license so the first run isn't interactive.
os.environ.setdefault("COQUI_TOS_AGREED", "1")


def _wav_seconds(path: str) -> float:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="XTTS v2 voice-clone test")
    ap.add_argument("--ref", required=True, help="reference clip: 6-12s of clean speech (wav)")
    ap.add_argument("--text", default="Good evening, sir. All systems are online.")
    ap.add_argument("--out", default="voice/out.wav")
    ap.add_argument("--language", default="en")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    if not Path(args.ref).exists():
        raise SystemExit(
            f"reference clip not found: {args.ref}\n"
            "Drop a 6-12s clean voice clip there (see voice/README.md)."
        )

    import torch
    from TTS.api import TTS

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[xtts] device={device}")

    t0 = time.monotonic()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print(f"[xtts] model loaded in {time.monotonic() - t0:.1f}s")

    t1 = time.monotonic()
    tts.tts_to_file(
        text=args.text,
        speaker_wav=args.ref,
        language=args.language,
        file_path=args.out,
    )
    synth = time.monotonic() - t1

    audio_s = _wav_seconds(args.out)
    rtf = (synth / audio_s) if audio_s else 0.0
    print(
        f"[xtts] synthesized {len(args.text)} chars -> {args.out}\n"
        f"[xtts] synth={synth:.2f}s  audio={audio_s:.2f}s  RTF={rtf:.2f} "
        f"(lower is faster; <1.0 = faster than realtime)"
    )


if __name__ == "__main__":
    main()
