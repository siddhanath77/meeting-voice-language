from __future__ import annotations

import os
from typing import Optional

import numpy as np


def _try_import_rnnoise():
    try:
        # rnnoise is optional; if available, use it
        import rnnoise  # type: ignore

        return rnnoise
    except Exception:
        return None


class AudioDenoiser:
    """Simple denoiser with optional RNNoise fallback to a noise gate.

    Processes mono float32 frames in [-1, 1] of fixed frame_size.
    """

    def __init__(self, sample_rate: int, frame_size: int, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.channels = channels

        self._rnnoise = None
        if os.environ.get("ENABLE_RNNOISE", "1") == "1":
            self._rnnoise = _try_import_rnnoise()
            if self._rnnoise is not None:
                try:
                    self._rn_state = self._rnnoise.RNNoise()
                except Exception:
                    self._rnnoise = None

        # Noise gate thresholds
        self._gate_threshold = float(os.environ.get("NOISE_GATE_THRESHOLD", "0.015"))
        self._gate_attenuation = float(os.environ.get("NOISE_GATE_ATTENUATION", "0.2"))

    def process_frame(self, frame_mono_f32: np.ndarray) -> np.ndarray:
        if frame_mono_f32.dtype != np.float32:
            frame_mono_f32 = frame_mono_f32.astype(np.float32)

        if len(frame_mono_f32) != self.frame_size:
            # Pad or trim to frame_size for simplicity
            if len(frame_mono_f32) < self.frame_size:
                pad = np.zeros(self.frame_size - len(frame_mono_f32), dtype=np.float32)
                frame_mono_f32 = np.concatenate([frame_mono_f32, pad])
            else:
                frame_mono_f32 = frame_mono_f32[: self.frame_size]

        if self._rnnoise is not None:
            try:
                # rnnoise expects int16 PCM
                as_int16 = (np.clip(frame_mono_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
                denoised = self._rn_state.filter(as_int16)
                return (denoised.astype(np.float32) / 32768.0).astype(np.float32)
            except Exception:
                pass

        # Simple noise gate fallback
        rms = np.sqrt(np.mean(np.square(frame_mono_f32))) + 1e-8
        if rms < self._gate_threshold:
            return frame_mono_f32 * self._gate_attenuation
        return frame_mono_f32


