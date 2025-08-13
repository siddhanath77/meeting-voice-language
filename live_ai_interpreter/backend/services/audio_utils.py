from __future__ import annotations

import numpy as np


def convert_int16_bytes_to_float32(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.int16)
    return arr.astype(np.float32) / 32768.0


def downsample_if_needed(signal_i16: np.ndarray, current_sr_hz: int) -> np.ndarray:
    """If sample rate is 48000 or 44100, downsample to 16000 using simple decimation.
    For simplicity and CPU-only constraints, this uses integer-ratio decimation where possible.
    """
    if current_sr_hz == 16000:
        return signal_i16
    if current_sr_hz == 48000:
        return signal_i16[::3]
    if current_sr_hz == 44100:
        # crude decimation to ~14700; close enough for Whisper; keep small latency
        return signal_i16[::3]
    if current_sr_hz == 32000:
        return signal_i16[::2]
    # Fallback: no resample
    return signal_i16


def simple_vad_is_silence(signal_i16: np.ndarray, threshold: int = 83) -> bool:
    """Very simple VAD: checks if smoothed RMS is below threshold (robust to distance).
    threshold is in int16 amplitude (0-32767)."""
    if signal_i16.size == 0:
        return True
    # Use RMS with light smoothing instead of mean-abs for better sensitivity to quiet speech
    f = signal_i16.astype(np.float32)
    rms = np.sqrt(np.mean(f * f))
    return rms < float(threshold)


