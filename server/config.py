from __future__ import annotations

import os
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # Networking
    host: str = Field(default=os.environ.get("HOST", "0.0.0.0"))
    port: int = Field(default=int(os.environ.get("PORT", "5000")))

    # Audio
    default_sample_rate: int = Field(default=int(os.environ.get("DEFAULT_SAMPLE_RATE", "48000")))
    default_channels: int = Field(default=int(os.environ.get("DEFAULT_CHANNELS", "1")))
    default_frame_size: int = Field(default=int(os.environ.get("DEFAULT_FRAME_SIZE", "480")))

    # Features
    enable_rnnoise: bool = Field(default=os.environ.get("ENABLE_RNNOISE", "1") == "1")
    consent_required: bool = Field(default=os.environ.get("CONSENT_REQUIRED", "1") == "1")
    store_transcripts: bool = Field(default=os.environ.get("STORE_TRANSCRIPTS", "0") == "1")
    transcripts_path: str = Field(default=os.environ.get("TRANSCRIPTS_PATH", "/data/transcripts"))

    # Observability
    enable_metrics: bool = Field(default=os.environ.get("ENABLE_METRICS", "1") == "1")
    log_level: str = Field(default=os.environ.get("LOG_LEVEL", "INFO"))

    # Model controls (placeholders)
    model_precision: str = Field(default=os.environ.get("MODEL_PRECISION", "fp16"))  # fp32|fp16|int8
    asr_device: str = Field(default=os.environ.get("ASR_DEVICE", "cuda"))  # cpu|cuda
    mt_device: str = Field(default=os.environ.get("MT_DEVICE", "cuda"))
    tts_device: str = Field(default=os.environ.get("TTS_DEVICE", "cuda"))


settings = Settings()


