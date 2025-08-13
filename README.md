# meeting-voice-language

Real-time audio streaming stack: FastAPI WebSocket server and a browser demo client, designed for ASR → MT → TTS. Includes Docker/Kubernetes (GPU) scaffolding, monitoring, and load testing.

## A to Z Setup (Conda, Python 3.12)

You asked for full steps with Conda. Below is the exact flow on Windows (PowerShell) and works similarly on macOS/Linux shells.

1) Create and activate environment (Python 3.12)

```powershell
conda create -n meeting python=3.12 -y
conda activate meeting
```

2) Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3) Optional: Copy and edit environment variables

```powershell
Copy-Item example.env .env
# Edit .env if needed (PORT, DEFAULT_* langs, API keys, etc.)
```

## What to run (and what each module does)

There are two runnable servers, serving different purposes:

### A) Live AI Interpreter (full pipeline, CPU demo)
- Path: `live_ai_interpreter/backend/main.py`
- Does: Streams mic audio over WebSocket, runs ASR (Whisper CPU), MT (Marian CPU defaults), TTS (gTTS), and serves a simple web UI.
- Endpoints:
  - `GET /` — demo UI
  - `GET /static/*` — scripts/styles
  - `WS /ws/audio` — JSON control + audio streaming

Run:

```powershell
python -m uvicorn live_ai_interpreter.backend.main:app --host 0.0.0.0 --port 5000
# or
python live_ai_interpreter/backend/main.py
```

Open `http://localhost:5000` and use the Start/Stop buttons.

Message flow (client ↔ server):
- Client sends `start/config` JSON, then streams audio chunks
- Server emits `partial` and `final` transcript JSON
- Server streams TTS audio back for playback

### B) Minimal WS echo + denoise server (latency/loopback)
- Path: `server/main.py`
- Does: Accepts raw PCM frames over `WS /ws-audio`, applies a simple denoiser (RNNoise if available), and echoes audio back. Good for network and latency checks.
- Endpoints:
  - `WS /ws-audio` — Binary frames: header + PCM payload

Run:

```powershell
uvicorn server.main:APP --host 0.0.0.0 --port 5000
```

Test (k6 load script):

```powershell
k6 run -e WS_URL=ws://localhost:5000/ws-audio loadtest/audio_ws_test.js
```

Notes:
- `server/main.py` mounts a `client` folder at `/`, which is not included here. Use the Live AI Interpreter UI from section A for an in-browser demo, or test via scripts.

### C) Legacy sample (alternate CPU-only backend)
- Path: `backend/main.py`
- Does: A CPU-only prototype for ASR → MT → TTS streaming. It expects binary PCM frames on `WS /ws/audio` and sends JSON partial/final + binary MP3. It serves static files from `backend/static`, which may not exist by default.

Run:

```powershell
python backend/main.py
```

If you need a UI for this legacy server, prefer section A (they differ in WS message format). Otherwise, send raw PCM16 frames from your own client.

## Endpoints and protocols (at a glance)

- Live Interpreter (`live_ai_interpreter/backend/main.py`)
  - `GET /` UI, `WS /ws/audio` with JSON control and chunked audio streaming
  - Emits `partial`, `final` JSON and TTS audio for playback

- Minimal Server (`server/main.py`)
  - `WS /ws-audio` — binary: `<uint32 seq><float64 ts_ms><PCM s16le>`
  - Echoes denoised audio back with the same header for RTT measurement

## Docker

CPU image:

```bash
docker build -t meeting-voice-language:cpu -f Dockerfile .
docker run --rm -p 5000:5000 meeting-voice-language:cpu
```

GPU image (adjust CUDA per cluster):

```bash
docker build -t meeting-voice-language:gpu -f Dockerfile.gpu .
docker run --rm -p 5000:5000 --gpus all meeting-voice-language:gpu
```

## Kubernetes (GPU)

1) Ensure NVIDIA device plugin is installed
2) Apply manifests

```bash
kubectl apply -f k8s/
```

Prometheus scraping is enabled via pod annotations; optional `ServiceMonitor` is in `k8s/servicemonitor.yaml`.

## Observability
- Metrics endpoint is exposed via the FastAPI app if enabled by env (see `example.env`).
- For logs, see `observability/fluent-bit-configmap.yaml` (requires Elastic + Fluent Bit stack).

## Troubleshooting
- Windows + `uvloop/httptools`: These are skipped on Windows by markers in `requirements.txt`. That’s expected.
- Missing UI on `server/main.py`: Use the Live Interpreter UI (section A) or a custom client. The `client` folder is not included.
- RNNoise optional: On Linux with `rnnoise` installed, the minimal server will use it; otherwise, it falls back to a noise gate.

## Python version
- Recommended: Python 3.12.x (Conda example above)
- The package metadata enforces `python_requires>=3.12`.

## Load testing quick start

```bash
k6 run -e WS_URL=ws://localhost:5000/ws-audio loadtest/audio_ws_test.js
```

## GDPR
- Set `CONSENT_REQUIRED=1` to require client-side consent
- Set `STORE_TRANSCRIPTS=1` and mount `/data/transcripts` with encryption-at-rest for storage