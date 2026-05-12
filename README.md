# ai_platform_local

Local, end-to-end voice agent stack:

- **LiveKit Server** (local WebRTC SFU): `livekit_server/livekit-server.exe`
- **Frontend UI** (Next.js): `frontend/`
- **Python Agent** (STT → LLM → TTS glue): `src/voice_server.py` (run via `livekit_agent/src/agent.py`)

## One true “run it” (Windows / PowerShell)

From the repo root:

```powershell
.\scripts\dev.ps1
```

Then open:

- `http://localhost:3000` (frontend)

Click **Start call**.

To stop everything:

```powershell
.\scripts\stop.ps1
```

## Prerequisites

- Windows + PowerShell
- Python 3.10+ (recommended) + venv at `.venv`
- Node.js 18+ and `pnpm`

## First-time setup

### 1) Python dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### 2) Frontend dependencies

```powershell
cd .\frontend
# Recommended (matches lockfile)
pnpm install

# If you don't have pnpm, npm works too:
# npm install
cd ..
```

### 3) Environment files

- Backend env (agent reads this): `.env.local` (repo root)
- Frontend env: `frontend/.env.local`

If you don’t have them yet, copy:

- `livekit_agent/.env.example` → `.env.local` (repo root)
- `frontend/.env.example` → `frontend/.env.local`

Keep `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` consistent across both.

## How `scripts\dev.ps1` works

It starts **three processes**:

1. LiveKit server in `--dev` mode (defaults to `devkey/secret`)
2. Next.js frontend (`pnpm dev`)
3. Python agent worker (`python -m livekit_agent.src.agent dev`)

## Notes for shipping

- Do **not** commit real keys/secrets. Use `.env.example` + secret manager.
- Use a **32+ byte** `LIVEKIT_API_SECRET` (short secrets trigger JWT warnings).
- Pre-download Whisper models (avoid runtime downloads on first launch).
