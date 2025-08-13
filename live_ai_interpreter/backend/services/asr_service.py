from __future__ import annotations

"""Streaming-ish ASR with faster-whisper tiny/int8 on CPU.

We run short-window recognition repeatedly to generate partials.
"""

from typing import Optional, Tuple
import os
from io import BytesIO

import numpy as np
import requests
import soundfile as sf

try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:  # pragma: no cover - allow import on systems without the package yet
    WhisperModel = None  # type: ignore


class StreamingASR:
    def __init__(self, model_size: str = "tiny", compute_type: str = "int8") -> None:
        self.model_size = model_size
        self.compute_type = compute_type
        self._model: Optional[WhisperModel] = None
        self._use_openai = os.environ.get("ASR_PROVIDER", "local").lower() == "openai"
        self._openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self._openai_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self._openai_asr_model = os.environ.get("OPENAI_ASR_MODEL", "gpt-4o-mini-transcribe")

        if not self._use_openai and WhisperModel is not None:
            # CPU-only
            self._model = WhisperModel(model_size, device="cpu", compute_type=compute_type)

    def transcribe_window(self, audio_f32_mono: np.ndarray, sample_rate_hz: int) -> Tuple[str, Optional[str], bool]:
        """Return (text, language, confidence_is_high).

        We call faster-whisper in non-streaming mode on the window (e.g., last 6s), which is
        adequate for low-latency partials on CPU if we keep windows small.
        """
        if audio_f32_mono.size == 0:
            return "", None, False

        # OpenAI path
        if self._use_openai and self._openai_api_key:
            try:
                buf = BytesIO()
                sf.write(buf, audio_f32_mono, sample_rate_hz, format="WAV", subtype="PCM_16")
                buf.seek(0)
                url = f"{self._openai_base}/audio/transcriptions"
                headers = {"Authorization": f"Bearer {self._openai_api_key}"}
                files = {
                    "file": ("audio.wav", buf.getvalue(), "audio/wav"),
                }
                data = {"model": self._openai_asr_model}
                resp = requests.post(url, headers=headers, data=data, files=files, timeout=60)
                resp.raise_for_status()
                j = resp.json()
                text = j.get("text") or ""
                return text.strip(), None, bool(text)
            except Exception:
                # Fallback to local if available
                pass

        if self._model is None:
            # Lightweight fallback: simple energy-based stub so pipeline is visible when ASR is unavailable
            rms = float(np.sqrt(np.mean(np.square(audio_f32_mono))) + 1e-8)
            if rms > 0.01:
                return "(speaking)", None, False
            return "", None, False

        # faster-whisper expects float32 mono in range [-1,1]
        segments, info = self._model.transcribe(
            audio_f32_mono, language=None, vad_filter=False, beam_size=1, best_of=1
        )

        text_parts = []
        for seg in segments:
            if getattr(seg, "text", ""):
                text_parts.append(seg.text.strip())
        text = " ".join(tp for tp in text_parts if tp)
        return text.strip(), getattr(info, "language", None), True


