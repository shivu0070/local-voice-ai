# voice/ — XTTS v2 voice-clone experiment (JARVIS)

Standalone sandbox for trying **Coqui XTTS v2** voice cloning, *isolated from the main agent*.
Goal: clone a JARVIS voice from a short clip, hear the quality, and measure latency
**before** deciding whether to wire it into the live pipeline.

## Why standalone first (read this)

This dev box is a **4 GB RTX 3050**. XTTS v2 is ~1.8 GB and wants ~2–4 GB VRAM. Ollama
already lives on the GPU, so **XTTS + Ollama won't both fit** here — running both reignites
the VRAM-eviction problem we fixed. XTTS is also slower than Piper (~1–3 s/sentence vs ~0.7 s).

So:
- **Now:** test quality + latency here, offline. Decide if it's worth it.
- **Real-time JARVIS** belongs on the deployment box with a bigger GPU (≥8 GB), where XTTS +
  Ollama + Whisper all fit. The integration path is a new TTS backend alongside Piper.

## Setup (separate venv — do NOT use the agent's .venv)

```powershell
# from repo root
python -m venv voice\.venv-xtts
.\voice\.venv-xtts\Scripts\python -m pip install -U pip

# GPU (recommended): install CUDA torch FIRST, then coqui-tts
.\voice\.venv-xtts\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.\voice\.venv-xtts\Scripts\python -m pip install -r voice\requirements.txt
```

First run auto-downloads the XTTS v2 model (~1.8 GB) to your local TTS cache.

## Get a reference clip

Drop a **6–12 second** clean clip of the target voice (just speech, no music/SFX) at:

```
voice\refs\jarvis.wav
```

(A JARVIS clip from the films works; keep it local — it's gitignored, and movie audio is
copyrighted, so don't commit or publish it.)

## Test it

```powershell
.\voice\.venv-xtts\Scripts\python voice\clone_test.py --ref voice\refs\jarvis.wav --text "Good evening, sir. All systems are online."
```

It prints model-load time and synth time, and writes `voice\out.wav`. Listen, and check
whether the latency is acceptable for your use.

## License note

XTTS v2 ships under Coqui's **CPML (non-commercial)** license. Fine for a personal/portfolio
demo; not for commercial use.
