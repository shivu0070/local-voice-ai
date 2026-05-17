# Local Voice AI Assistant

https://github.com/user-attachments/assets/92743f6d-186e-42b4-b9c2-ff00c44f2c8f

A fully-local, real-time voice assistant. Speak to it, it speaks back — no cloud APIs, no external services, everything runs on a single PC.

**Stack:**
- **LiveKit Server** (local WebRTC SFU) — `livekit_server/livekit-server.exe`
- **faster-whisper** (STT) — `large-v3-turbo` on CUDA
- **Silero VAD** — speech-onset detection + barge-in
- **Ollama** (LLM) — local models, e.g. `qwen2.5:0.5b` or `llama3.2:3b`
- **Piper** (TTS) — `bin/piper.exe` with ONNX voices
- **Next.js** frontend — `frontend/`
- **Python agent** glue — `src/voice_server.py`, launched via `livekit_agent/src/agent.py`

**Features:**
- Sub-second barge-in (Silero VAD cuts off TTS within ~one word of you speaking)
- TTS prefetch pipeline (synthesizes the next chunk while the current one plays — no inter-chunk gaps)
- Streaming LLM → sentence-by-sentence TTS so audio starts as soon as the first clause is generated
- Two brain modes: `chat` (streaming) and `router` (JSON tool-calls)

## Prerequisites

- Windows + PowerShell (works on other platforms with path tweaks)
- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com) running locally with at least one model pulled (`ollama pull qwen2.5:0.5b`)
- An NVIDIA GPU strongly recommended (for Whisper + Ollama)

## First-time setup

### 1) Python dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### 2) Frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

### 3) Models

Models aren't committed (they're large). Download:

- **Piper voice:** drop an `.onnx` + matching `.onnx.json` into `models/piper/` (default config expects `en_US-danny-low`). Voices: https://github.com/rhasspy/piper/blob/master/VOICES.md
- **Whisper model:** `large-v3-turbo` from Hugging Face. The agent will fetch it on first run if not present, into `models/hub/`.

### 4) Environment

Copy the example config and you're done — the defaults match the local LiveKit dev key, so no edits are needed for a local run:

```powershell
copy .env.example .env.local
```

`.env.local` is gitignored (it's where your runtime config lives). `.env.example` is the committed reference everyone clones.

## Run it

Open **three terminals** in the repo root and run one command in each. They all need to stay running.

**Terminal 1 — LiveKit server:**
```powershell
.\livekit_server\livekit-server.exe --dev --keys "devkey: 6f1d0c9b7a34e6c2d8f501c9a4b3e2975c4d8f6a1b2c3d4e5f60718293a0b1c0"
```

**Terminal 2 — Frontend:**
```powershell
cd frontend
npm run dev
```

**Terminal 3 — Python agent:**
```powershell
python -m livekit_agent.src.agent dev
```

Then open **http://localhost:3000** in your browser and click **Start call**. Talk to it.

## Config knobs (`.env.local`)

| Key | Effect |
|---|---|
| `OLLAMA_MODEL` | Which Ollama model to use. `qwen2.5:0.5b` for fast, `llama3.2:3b-instruct-q4_0` for better quality. |
| `CHAT_NUM_PREDICT` | Max tokens per reply. Cap to keep answers short and responses snappy. |
| `WHISPER_MODEL` / `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` | Whisper config. `int8_float16` on CUDA balances speed and VRAM. |
| `WHISPER_BEAM_SIZE` | `1` for greedy decoding (fastest); higher = slightly more accurate, slower. |
| `SILERO_DEVICE` | Keep on `cpu` — Silero VAD is tiny and GPU round-trip per chunk makes it *slower*. |
| `SILERO_MIN_SILENCE_MS` | How long you must pause before the bot treats you as done speaking. Lower = snappier, more interrupts you mid-thought. |
| `BARGE_MIN_SEC` | How long sustained speech is needed to interrupt TTS. Default 0.2s (~one word). Raise if background noise stops the bot. |
| `PIPER_MODEL` / `PIPER_CONFIG` | Which Piper voice to use. |

## How a turn flows

```
mic audio → LiveKit room → AgentSession
                          ├─ Silero VAD ──► barge-in detection
                          └─ StreamAdapter
                              ├─ Silero VAD ──► endpointing
                              └─ faster-whisper ──► transcript
                                                   │
                                                   ▼
                                       Ollama (streaming tokens)
                                                   │
                                       sentence chunks
                                                   ▼
                                       Piper TTS (prefetched 1 ahead)
                                                   │
                                                   ▼
                                          LiveKit ──► speaker
```

## Design decisions

A few non-obvious choices in this codebase and the reasoning behind each — every one of these started as a measurable latency problem and ended in a concrete fix.

**TTS synthesis pipelined with playback, not serialized.** Naively, `synthesize(N) → play(N) → synthesize(N+1) → play(N+1)` leaves a ~150–400 ms silence between every chunk (Piper subprocess spawn + ONNX model load). The speaker worker keeps one synthesis task in flight while the current chunk plays out, so the next chunk's audio is ready before the current one finishes. Result: no inter-chunk gaps, no buffer underruns, smooth speech.

**Silero VAD runs on CPU, not the GPU it could.** With both Whisper and Silero on CUDA, Silero fell 0.6–2.8 s behind realtime — the per-chunk GPU round-trip cost more than the inference itself, and it was competing with Whisper for the device. Silero is small enough that a single CPU core handles it faster than realtime. Counterintuitive but visible in the latency logs.

**A separate VAD on `AgentSession` for barge-in.** The StreamAdapter wrapping Whisper has its own VAD for STT segmentation, but `AgentSession(vad=None)` meant the session had no fast speech detector and waited for the *final* transcript (≥ 650 ms silence + decode) before cutting off TTS. Giving the session its own VAD plus `min_interruption_duration=0.2 s` brings TTS cut-off down to roughly one word of user speech.

**Single-chunk synthesis lookahead, not deeper.** Buffering more than one chunk ahead just makes barge-in messier — every queued chunk has to be cancelled on interrupt. One ahead is enough to hide synth latency under playback, no more.

**Early first-clause flush.** Sentence-by-sentence TTS would wait for the first `.!?` before speaking; for long opening sentences that's seconds of dead air. The token loop flushes the opening clause at the first comma/colon after ~50 characters, then falls back to sentence boundaries. Audio starts the moment the first thought has formed.

**No manual frame pacing.** An earlier version paced the audio generator with `await asyncio.sleep(20 ms)` per 20 ms frame, but Windows' timer granularity is ~15 ms and the slop caused intermittent buffer underruns (audible glitches). LiveKit's `AudioSource` already buffers and paces at exact realtime with backpressure, so the generator now just yields and trusts it.

