from __future__ import annotations

"""gTTS-based synthesis to MP3 bytes.

Note: gTTS uses Google's TTS HTTP API, so this requires outbound internet.
For fully offline, replace with pyttsx3 or an on-device TTS model.
"""

import os
from io import BytesIO
import requests
from gtts import gTTS  # type: ignore


class TextToSpeech:
    def __init__(self, default_lang: str = "en") -> None:
        self.default_lang = default_lang
        self._use_openai = os.environ.get("TTS_PROVIDER", "local").lower() == "openai"
        self._openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self._openai_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self._openai_tts_model = os.environ.get("OPENAI_TTS_MODEL", "tts-1")

    def synthesize_to_mp3(self, text: str, lang: str | None = None) -> bytes:
        """Return MP3 bytes for given text."""
        # OpenAI path
        if self._use_openai and self._openai_api_key:
            try:
                url = f"{self._openai_base}/audio/speech"
                headers = {
                    "Authorization": f"Bearer {self._openai_api_key}",
                    "Content-Type": "application/json",
                }
                voice = "alloy"  # default voice id
                body = {
                    "model": self._openai_tts_model,
                    "voice": voice,
                    "input": text,
                    "format": "mp3",
                }
                resp = requests.post(url, headers=headers, json=body, timeout=60)
                resp.raise_for_status()
                return resp.content
            except Exception:
                pass

        lang_code = (lang or self.default_lang)[:2]
        buf = BytesIO()
        tts = gTTS(text=text, lang=lang_code)
        tts.write_to_fp(buf)
        return buf.getvalue()


