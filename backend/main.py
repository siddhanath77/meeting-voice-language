from __future__ import annotations

"""
Live AI Interpreter - CPU-only, Windows 10, Python 3.12 compatible.

Features:
- FastAPI app (port 5000)
- WebSocket /ws/audio for low-latency audio streaming
- Streaming-ish ASR with faster-whisper (tiny, int8, CPU)
- Translation with MarianMT (Hindi→English by default)
- TTS with gTTS (English), returned as small MP3 clips
- Static frontend served at '/'
- Health check at '/health'

Run:  python backend/main.py
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Deque, Optional, Tuple
from collections import deque
import contextlib

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Local imports (support both `python backend/main.py` and module execution)
try:
    from .config import Settings  # type: ignore
    from .services.audio_utils import (
        downsample_if_needed,
        simple_vad_is_silence,
    )
    from .services.asr_service import StreamingASR
    from .services.mt_service import Translator
    from .services.tts_service import TextToSpeech
except Exception:  # pragma: no cover
    # Fallback: adjust sys.path to project root to allow direct script run
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from backend.config import Settings  # type: ignore
    from backend.services.audio_utils import (  # type: ignore
        downsample_if_needed,
        simple_vad_is_silence,
    )
    from backend.services.asr_service import StreamingASR  # type: ignore
    from backend.services.mt_service import Translator  # type: ignore
    from backend.services.tts_service import TextToSpeech  # type: ignore


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


settings = Settings()

app = FastAPI(title="Live AI Interpreter (CPU)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve static frontend
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=False), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


class ConnectionPipelines:
    """Holds per-connection state and pipelines for ASR/MT/TTS."""

    def __init__(self, input_sample_rate_hz: int = 16000) -> None:
        self.input_sample_rate_hz = input_sample_rate_hz
        self.buffer_int16: Deque[np.ndarray] = deque(maxlen=600)  # ~12s at 20ms frames

        # Pipelines
        self.asr = StreamingASR(model_size="tiny", compute_type="int8")
        self.translator = Translator(default_src="hi", default_tgt="en")
        self.tts = TextToSpeech(default_lang="en")

        # State for streaming UX
        self._last_partial_text: str = ""
        self._pending_final_text: str = ""
        self._closed: bool = False

    def close(self) -> None:
        self._closed = True

    def append_int16_frame(self, frame: np.ndarray) -> None:
        self.buffer_int16.append(frame)

    def get_recent_audio(self, seconds: float = 6.0) -> Tuple[np.ndarray, int]:
        """Return a mono int16 numpy array of the last N seconds and its sample rate."""
        samples_needed = int(seconds * self.input_sample_rate_hz)
        collected: list[np.ndarray] = []
        total = 0
        for arr in reversed(self.buffer_int16):
            collected.append(arr)
            total += len(arr)
            if total >= samples_needed:
                break
        if not collected:
            return np.zeros(0, dtype=np.int16), self.input_sample_rate_hz
        audio = np.concatenate(list(reversed(collected)))
        if len(audio) > samples_needed:
            audio = audio[-samples_needed:]
        return audio, self.input_sample_rate_hz


async def asr_mt_tts_loop(websocket: WebSocket, conn: ConnectionPipelines) -> None:
    """Background coroutine that periodically runs ASR, sends partials, and emits final+TTS when silence occurs."""
    try:
        while not conn._closed:
            await asyncio.sleep(0.7)  # latency/CPU trade-off

            audio_i16, sr = conn.get_recent_audio(seconds=6.0)
            if len(audio_i16) < int(1.0 * sr):
                continue  # need at least ~1s to say anything meaningful

            # Convert to float32 mono in [-1,1]
            audio_f32 = audio_i16.astype(np.float32) / 32768.0

            # Run ASR on the last window
            partial_text, language, _ = conn.asr.transcribe_window(audio_f32, sr)
            if not partial_text:
                continue

            # Only send if changed to reduce chatter
            if partial_text != conn._last_partial_text:
                conn._last_partial_text = partial_text
                # Translate partial as well for UX
                try:
                    translated_part, tgt_lang_part = conn.translator.translate(partial_text, src_lang=language)
                except Exception:
                    translated_part, tgt_lang_part = "", settings.default_tgt_lang
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "partial",
                            "text": partial_text,
                            "lang": language or "",
                            "translation": translated_part,
                            "target_lang": tgt_lang_part,
                        }
                    )
                )

            # VAD on last 0.6s to detect pause → finalize
            tail = audio_i16[-int(0.6 * sr) :]
            if simple_vad_is_silence(tail, threshold=400):  # threshold in int16 amplitude
                if partial_text.strip():
                    final_text = partial_text.strip()
                    conn._pending_final_text = ""
                    conn._last_partial_text = ""

                    # Translate
                    translated, tgt_lang = conn.translator.translate(final_text, src_lang=language)

                    # Send final text
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "final",
                                "text": final_text,
                                "lang": language or "",
                                "translation": translated,
                                "target_lang": tgt_lang,
                            }
                        )
                    )

                    # Synthesize and send one MP3 clip (small chunk)
                    mp3_bytes = await asyncio.get_event_loop().run_in_executor(
                        None, conn.tts.synthesize_to_mp3, translated, tgt_lang
                    )
                    # Send as a single binary WS frame; client will play immediately
                    await websocket.send_bytes(mp3_bytes)
    except WebSocketDisconnect:
        return
    except Exception:
        # Avoid crashing background task; errors will end with connection close
        pass


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket) -> None:
    await websocket.accept()

    # Expect a config message first, otherwise default to 16k
    input_sample_rate = 16000
    frame_size = 320  # 20ms @16k

    conn = ConnectionPipelines(input_sample_rate_hz=input_sample_rate)
    bg_task = asyncio.create_task(asr_mt_tts_loop(websocket, conn))

    try:
        while True:
            msg = await websocket.receive()
            if "text" in msg and msg["text"]:
                try:
                    payload = json.loads(msg["text"])  # type: ignore[index]
                except Exception:
                    continue
                if payload.get("type") == "config":
                    input_sample_rate = int(payload.get("sampleRate", 16000))
                    frame_size = int(payload.get("frameSize", 320))
                    conn.input_sample_rate_hz = input_sample_rate
                    await websocket.send_text(json.dumps({"type": "ack"}))
                continue

            data: Optional[bytes] = msg.get("bytes") if isinstance(msg, dict) else None  # type: ignore[assignment]
            if not data:
                await asyncio.sleep(0)
                continue

            # Assume client sends mono s16le PCM at the declared sample rate
            # For safety, downsample if it's 48000→16000 in 3:1 ratio
            in_i16 = np.frombuffer(data, dtype=np.int16)
            in_i16 = downsample_if_needed(in_i16, current_sr_hz=input_sample_rate)

            # Split into fixed frames to aid buffering consistency
            if len(in_i16) < frame_size:
                conn.append_int16_frame(in_i16)
            else:
                for i in range(0, len(in_i16), frame_size):
                    conn.append_int16_frame(in_i16[i : i + frame_size])

    except WebSocketDisconnect:
        pass
    finally:
        conn.close()
        try:
            if not bg_task.done():
                bg_task.cancel()
                with contextlib.suppress(Exception):
                    await bg_task
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    # Allow direct script run: module path may be unknown, so pass app object directly
    uvicorn.run(app, host="0.0.0.0", port=5000)
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import rest_api, ws_audio


app = FastAPI(title="Meeting Voice Language")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rest_api.router)
app.include_router(ws_audio.router)


def create_app() -> FastAPI:
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.host, port=settings.port)


