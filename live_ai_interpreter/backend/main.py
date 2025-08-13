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
import logging
from logging.handlers import RotatingFileHandler
import os
import time
import re
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
        convert_int16_bytes_to_float32,  # noqa: F401 - kept for future use
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
        convert_int16_bytes_to_float32,  # noqa: F401
        downsample_if_needed,
        simple_vad_is_silence,
    )
    from backend.services.asr_service import StreamingASR  # type: ignore
    from backend.services.mt_service import Translator  # type: ignore
    from backend.services.tts_service import TextToSpeech  # type: ignore


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


LOG_LEVEL_NAME = os.environ.get("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("lai")
logger.setLevel(getattr(logging, LOG_LEVEL_NAME, logging.INFO))

# Configure console + file logging with rotation
if not logger.handlers:
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    # Console
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setLevel(getattr(logging, LOG_LEVEL_NAME, logging.INFO))
    _handler.setFormatter(fmt)
    logger.addHandler(_handler)

    # File (logs/app.log by default)
    try:
        LOG_DIR = os.environ.get("LOG_DIR", "logs")
        LOG_FILE_NAME = os.environ.get("LOG_FILE", "app.log")
        LOG_MAX_BYTES = int(os.environ.get("LOG_MAX_BYTES", str(5 * 1024 * 1024)))  # 5MB
        LOG_BACKUP_COUNT = int(os.environ.get("LOG_BACKUP_COUNT", "5"))

        os.makedirs(LOG_DIR, exist_ok=True)
        file_path = os.path.join(LOG_DIR, LOG_FILE_NAME)
        file_handler = RotatingFileHandler(
            file_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, LOG_LEVEL_NAME, logging.INFO))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except Exception:
        # If file logging fails, continue with console logging only
        pass
logger.propagate = False

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
        # Simple per-frame AGC to help low-volume voices
        try:
            if frame.dtype != np.int16:
                frame = frame.astype(np.int16)
            # root-mean-square in int16 units
            rms = float(np.sqrt(np.mean(np.square(frame.astype(np.float32)))) + 1e-8)
            target_rms = 2000.0  # ~0.061 in float32; modest gain target
            if rms > 0:
                gain = min(10.0, target_rms / rms)
                if gain > 1.0:
                    amplified = np.clip((frame.astype(np.float32) * gain), -32768.0, 32767.0).astype(np.int16)
                    self.buffer_int16.append(amplified)
                    return
        except Exception:
            pass
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


class RoomManager:
    def __init__(self) -> None:
        self._rooms: dict[str, set[WebSocket]] = {}

    def register(self, room_id: str, ws: WebSocket) -> None:
        self._rooms.setdefault(room_id, set()).add(ws)

    def unregister(self, ws: WebSocket) -> None:
        empty_rooms: list[str] = []
        for rid, members in self._rooms.items():
            if ws in members:
                members.discard(ws)
            if not members:
                empty_rooms.append(rid)
        for rid in empty_rooms:
            self._rooms.pop(rid, None)

    def peers(self, room_id: str, ws: WebSocket) -> list[WebSocket]:
        members = self._rooms.get(room_id) or set()
        return [m for m in members if m is not ws]


ROOMS = RoomManager()


class TranscriptHistory:
    def __init__(self, retention_secs: int = 1800) -> None:
        self._retention_secs = retention_secs
        self._rooms: dict[str, deque[dict]] = {}

    def add_final(self, room_id: str, text: str, lang: str | None, translation: str, target_lang: str | None) -> None:
        now = time.time()
        entry = {
            "ts": now,
            "type": "final",
            "text": text,
            "lang": lang or "",
            "translation": translation,
            "target_lang": target_lang or "",
        }
        dq = self._rooms.setdefault(room_id, deque())
        dq.append(entry)
        self._cleanup(room_id, now)

    def get_recent(self, room_id: str) -> list[dict]:
        now = time.time()
        self._cleanup(room_id, now)
        return list(self._rooms.get(room_id, deque()))

    def _cleanup(self, room_id: str, now_ts: float) -> None:
        dq = self._rooms.get(room_id)
        if not dq:
            return
        # Drop anything older than retention window
        cutoff = now_ts - float(self._retention_secs)
        while dq and dq[0].get("ts", now_ts) < cutoff:
            dq.popleft()


HISTORY_RETENTION_SECS = int(os.environ.get("HISTORY_RETENTION_SECS", "1800"))
HISTORY = TranscriptHistory(retention_secs=HISTORY_RETENTION_SECS)


async def asr_mt_tts_loop(websocket: WebSocket, conn: ConnectionPipelines, room_id: str) -> None:
    """Background coroutine that periodically runs ASR, sends partials, and emits final+TTS when silence occurs."""
    try:
        while not conn._closed:
            await asyncio.sleep(0.25)  # faster partial cadence

            audio_i16, sr = conn.get_recent_audio(seconds=1.2)
            if len(audio_i16) < int(0.6 * sr):
                continue  # need at least ~1s to say anything meaningful

            # Convert to float32 mono in [-1,1]
            audio_f32 = audio_i16.astype(np.float32) / 32768.0

            # Run ASR on the last window (offload blocking work to thread pool)
            loop = asyncio.get_running_loop()
            partial_text, language, is_confident = await loop.run_in_executor(
                None, conn.asr.transcribe_window, audio_f32, sr
            )
            if not partial_text:
                continue

            # Only send if changed to reduce chatter
            if partial_text != conn._last_partial_text:
                conn._last_partial_text = partial_text
                logger.debug("partial text=%r lang=%s room=%s", partial_text[:64], language, room_id)
                # Translate partial as well for UX
                try:
                    translated_part, tgt_lang_part = await loop.run_in_executor(
                        None, conn.translator.translate, partial_text, language, None
                    )
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
            tail = audio_i16[-int(0.2 * sr) :]
            if simple_vad_is_silence(tail, threshold=90):  # more sensitive finalize
                if partial_text.strip():
                    final_text = partial_text.strip()
                    conn._pending_final_text = ""
                    conn._last_partial_text = ""

                    # Translate (offloaded)
                    translated, tgt_lang = await loop.run_in_executor(
                        None, conn.translator.translate, final_text, language, None
                    )

                    # Send final text
                    logger.info(
                        "final text len=%d lang=%s tgt=%s room=%s",
                        len(final_text), language, tgt_lang, room_id,
                    )
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

                    # Save to in-memory history (30 min by default)
                    try:
                        HISTORY.add_final(room_id, final_text, language, translated, tgt_lang)
                    except Exception:
                        pass

                    # Low-latency TTS: send first short chunk immediately, then the rest if applicable
                    first_seg = translated
                    remaining = ""
                    try:
                        # split on sentence enders, fallback to a fixed prefix
                        m = re.search(r"([.!?])", translated)
                        if m and m.start() + 1 < len(translated):
                            idx = m.end()
                            first_seg = translated[:idx]
                            remaining = translated[idx:].strip()
                        elif len(translated) > 60:
                            first_seg = translated[:60]
                            remaining = translated[60:]
                    except Exception:
                        pass

                    loop = asyncio.get_running_loop()
                    mp3_bytes_first = await loop.run_in_executor(
                        None, conn.tts.synthesize_to_mp3, first_seg, tgt_lang
                    )
                    logger.debug("tts first bytes=%d room=%s", len(mp3_bytes_first or b""), room_id)
                    await websocket.send_bytes(mp3_bytes_first)

                    if remaining:
                        mp3_bytes_rest = await loop.run_in_executor(
                            None, conn.tts.synthesize_to_mp3, remaining, tgt_lang
                        )
                        logger.debug("tts rest bytes=%d room=%s", len(mp3_bytes_rest or b""), room_id)
                        await websocket.send_bytes(mp3_bytes_rest)

                    # Broadcast to room peers (best-effort)
                    for peer in ROOMS.peers(room_id, websocket):
                        try:
                            await peer.send_text(
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
                            await peer.send_bytes(mp3_bytes_first)
                            if remaining:
                                await peer.send_bytes(mp3_bytes_rest)
                        except Exception as exc:
                            logger.debug("broadcast failure: %s", exc)
                            # Ignore peer send failures
                            pass
    except asyncio.CancelledError:
        return
    except WebSocketDisconnect:
        return
    except Exception:
        # Avoid crashing background task; errors will end with connection close
        logger.exception("error in asr_mt_tts_loop")


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("ws connect from client")

    # Expect a config message first, otherwise default to 16k
    input_sample_rate = 16000
    frame_size = 320  # 20ms @16k
    room_id = "default"

    conn = ConnectionPipelines(input_sample_rate_hz=input_sample_rate)
    bg_task: Optional[asyncio.Task] = None
    last_audio_ts: float = 0.0
    last_audio_log_ts: float = 0.0
    warned_no_audio: bool = False

    try:
        while True:
            try:
                msg = await websocket.receive()
            except WebSocketDisconnect:
                break
            except RuntimeError:
                # Starlette raises when receive() is called after disconnect
                break
            if "text" in msg and msg["text"]:
                try:
                    payload = json.loads(msg["text"])  # type: ignore[index]
                except Exception:
                    continue
                if payload.get("type") == "config":
                    input_sample_rate = int(payload.get("sampleRate", 16000))
                    frame_size = int(payload.get("frameSize", 320))
                    target_lang = payload.get("targetLang")
                    room_id = str(payload.get("roomId", room_id or "default"))
                    logger.info(
                        "config sr=%d frame=%d tgt=%s room=%s",
                        input_sample_rate, frame_size, target_lang, room_id,
                    )
                    conn.input_sample_rate_hz = input_sample_rate
                    # pass target lang override via translator default if provided
                    if target_lang:
                        conn.translator.default_tgt = str(target_lang)
                    # Register to room and start background loop once configured
                    try:
                        ROOMS.register(room_id, websocket)
                    except Exception:
                        pass
                    if bg_task is None:
                        bg_task = asyncio.create_task(asr_mt_tts_loop(websocket, conn, room_id))
                    await websocket.send_text(json.dumps({"type": "ack"}))
                elif payload.get("type") == "heartbeat":
                    ts = payload.get("ts_ms")
                    logger.debug("heartbeat from client ts=%s room=%s", ts, room_id)
                    await websocket.send_text(json.dumps({"type": "pong", "ts_ms": ts}))
                elif payload.get("type") == "history_request":
                    # Send recent transcript history for this room
                    try:
                        items = HISTORY.get_recent(room_id)
                    except Exception:
                        items = []
                    await websocket.send_text(json.dumps({"type": "history", "items": items}))
                continue

            data: Optional[bytes] = msg.get("bytes") if isinstance(msg, dict) else None  # type: ignore[assignment]
            if not data:
                # Periodically warn if no audio arriving
                now = time.time()
                if not warned_no_audio and last_audio_ts == 0.0 and (now - last_audio_log_ts) > 3.0:
                    try:
                        await websocket.send_text(json.dumps({"type": "warning", "code": "NO_AUDIO", "message": "No audio received yet. Check mic permission/input device."}))
                    except Exception:
                        pass
                    warned_no_audio = True
                    last_audio_log_ts = now
                await asyncio.sleep(0)
                continue

            # Assume client sends mono s16le PCM at the declared sample rate
            # For safety, downsample if it's 48000→16000 in 3:1 ratio
            in_i16 = np.frombuffer(data, dtype=np.int16)
            in_i16 = downsample_if_needed(in_i16, current_sr_hz=input_sample_rate)
            last_audio_ts = time.time()
            # Log audio stats roughly once per second
            if (last_audio_ts - last_audio_log_ts) > 1.0:
                try:
                    rms = float(np.sqrt(np.mean((in_i16.astype(np.float32)) ** 2)))
                    logger.debug("rx bytes=%d frames=%d rms=%.1f sr=%d", len(data or b""), len(in_i16), rms, input_sample_rate)
                except Exception:
                    pass
                last_audio_log_ts = last_audio_ts

            # Split into fixed frames to aid buffering consistency
            if len(in_i16) < frame_size:
                conn.append_int16_frame(in_i16)
            else:
                for i in range(0, len(in_i16), frame_size):
                    conn.append_int16_frame(in_i16[i : i + frame_size])

    except WebSocketDisconnect:
        logger.info("ws disconnect")
        pass
    finally:
        conn.close()
        try:
            if bg_task is not None and not bg_task.done():
                bg_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await bg_task
        except Exception:
            pass
        try:
            ROOMS.unregister(websocket)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    # Allow direct script run: module path may be unknown, so pass app object directly
    uvicorn.run(app, host="0.0.0.0", port=5000)


