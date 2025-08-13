from __future__ import annotations

import asyncio
import struct
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from pathlib import Path

from .processing import AudioDenoiser


APP = FastAPI()

APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve static client files from ../client
_SERVER_DIR = Path(__file__).resolve().parent
_CLIENT_DIR = (_SERVER_DIR.parent / "client").resolve()
APP.mount("/", StaticFiles(directory=str(_CLIENT_DIR), html=True), name="static")


HEADER_STRUCT = struct.Struct("<Id")  # uint32 seq, float64 timestamp_ms


class ConnectionState:
    def __init__(self) -> None:
        self.sample_rate: int = 48000
        self.channels: int = 1
        self.frame_size: int = 480  # samples per frame (10ms at 48k)
        self.format: str = "s16le"
        self.denoiser: Optional[AudioDenoiser] = None


@APP.websocket("/ws-audio")
async def ws_audio(websocket: WebSocket) -> None:
    await websocket.accept()
    state = ConnectionState()
    state.denoiser = AudioDenoiser(
        sample_rate=state.sample_rate,
        frame_size=state.frame_size,
        channels=state.channels,
    )

    try:
        while True:
            message = await websocket.receive()

            if "text" in message and message["text"] is not None:
                # Expect JSON config on first message
                import json

                try:
                    payload = json.loads(message["text"])  # type: ignore[index]
                except Exception:
                    await websocket.send_text("{\"type\":\"error\",\"message\":\"invalid_json\"}")
                    continue

                if payload.get("type") == "config":
                    state.sample_rate = int(payload.get("sampleRate", 48000))
                    state.channels = int(payload.get("channelCount", 1))
                    state.frame_size = int(payload.get("frameSize", 480))
                    state.format = str(payload.get("format", "s16le"))
                    state.denoiser = AudioDenoiser(
                        sample_rate=state.sample_rate,
                        frame_size=state.frame_size,
                        channels=state.channels,
                    )
                    await websocket.send_text("{\"type\":\"ack\"}")
                else:
                    await websocket.send_text("{\"type\":\"error\",\"message\":\"unknown_text_message\"}")

                continue

            data: Optional[bytes] = message.get("bytes") if isinstance(message, dict) else None  # type: ignore[assignment]
            if not data:
                await asyncio.sleep(0)  # yield
                continue

            if len(data) < HEADER_STRUCT.size:
                # Ignore malformed frames
                continue

            seq, ts_ms = HEADER_STRUCT.unpack_from(data, 0)
            pcm_bytes = data[HEADER_STRUCT.size :]

            if state.format != "s16le":
                # Only s16le supported in this minimal example
                continue

            # Convert to float32 in [-1, 1]
            pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            if state.channels != 1:
                # Downmix to mono
                pcm_int16 = pcm_int16.reshape(-1, state.channels).mean(axis=1).astype(np.int16)

            # Apply denoise per frame_size
            frame = pcm_int16.astype(np.float32) / 32768.0

            if len(frame) % state.frame_size != 0:
                # Trim to whole frames for processing
                usable_len = (len(frame) // state.frame_size) * state.frame_size
                frame = frame[:usable_len]

            denoised_frames: list[np.ndarray] = []
            if state.denoiser is not None:
                for i in range(0, len(frame), state.frame_size):
                    chunk = frame[i : i + state.frame_size]
                    denoised = state.denoiser.process_frame(chunk)
                    denoised_frames.append(denoised)
            else:
                denoised_frames.append(frame)

            if denoised_frames:
                out = np.concatenate(denoised_frames)
            else:
                out = frame

            out = np.clip(out, -1.0, 1.0)
            out_int16 = (out * 32767.0).astype(np.int16)

            # Echo back with same header for RTT measurement
            out_bytes = HEADER_STRUCT.pack(seq, float(ts_ms)) + out_int16.tobytes()
            await websocket.send_bytes(out_bytes)

    except WebSocketDisconnect:
        return
    except Exception:
        # Do not crash server on processing errors
        try:
            await websocket.close()
        except Exception:
            pass


def create_app() -> FastAPI:
    return APP


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:APP", host="0.0.0.0", port=5000, reload=False)


