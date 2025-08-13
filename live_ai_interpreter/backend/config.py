from __future__ import annotations

from pydantic import BaseModel


class Settings(BaseModel):
    port: int = 5000
    default_src_lang: str = "hi"  # Hindi by default
    default_tgt_lang: str = "en"
    asr_model_size: str = "tiny"
    asr_compute_type: str = "int8"


