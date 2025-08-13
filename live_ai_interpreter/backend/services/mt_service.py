from __future__ import annotations

"""Minimal MarianMT translation for CPU.

Defaults to Hindi→English using Helsinki-NLP opus-mt models.
"""

from typing import Tuple
import os
import requests

try:
    from transformers import MarianMTModel, MarianTokenizer  # type: ignore
    import torch  # type: ignore
except Exception:  # pragma: no cover
    MarianMTModel = None  # type: ignore
    MarianTokenizer = None  # type: ignore
    torch = None  # type: ignore


class Translator:
    def __init__(self, default_src: str = "hi", default_tgt: str = "en") -> None:
        self.default_src = default_src
        self.default_tgt = default_tgt
        self._models = {}
        self._tokenizers = {}
        self._use_openai = os.environ.get("MT_PROVIDER", "local").lower() == "openai"
        self._openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self._openai_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self._openai_mt_model = os.environ.get("OPENAI_MT_MODEL", "gpt-4o-mini")

    def _load_pair(self, src: str, tgt: str) -> Tuple[object, object]:
        pair = f"{src}-{tgt}"
        if pair in self._models:
            return self._models[pair], self._tokenizers[pair]

        if MarianMTModel is None or MarianTokenizer is None:
            raise RuntimeError("transformers not installed")

        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        tok = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        if torch is not None:
            model.to("cpu")
            model.eval()
        self._models[pair] = model
        self._tokenizers[pair] = tok
        return model, tok

    def translate(self, text: str, src_lang: str | None, tgt_override: str | None = None) -> Tuple[str, str]:
        src = src_lang or self.default_src
        tgt = tgt_override or self.default_tgt
        # OpenAI path
        if self._use_openai and self._openai_api_key:
            try:
                url = f"{self._openai_base}/chat/completions"
                headers = {"Authorization": f"Bearer {self._openai_api_key}", "Content-Type": "application/json"}
                prompt = (
                    f"Translate the following text from {src} to {tgt}. "
                    f"Only return the translation without extra commentary.\n\n{text}"
                )
                body = {"model": self._openai_mt_model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
                resp = requests.post(url, headers=headers, json=body, timeout=60)
                resp.raise_for_status()
                j = resp.json()
                out = (j.get("choices") or [{}])[0].get("message", {}).get("content", "")
                return out.strip(), tgt
            except Exception:
                # fall through to local
                pass

        try:
            model, tok = self._load_pair(src, tgt)
            inputs = tok([text], return_tensors="pt", padding=True)
            if torch is not None:
                with torch.no_grad():
                    generated = model.generate(**inputs, max_length=128, num_beams=1)
            else:
                generated = model.generate(**inputs, max_length=128, num_beams=1)
            out = tok.batch_decode(generated, skip_special_tokens=True)
            cleaned = out[0].strip() if out else ""
            return (cleaned or text), tgt
        except Exception as exc:
            # Fallback: echo source when translation fails
            return text, tgt


