"""Translation through Helsinki-NLP/opus-mt-tc-bible-big-inc-en.

Covers ~20 Indic languages including Sylheti (syl), trained on OPUS +
Bible-translation corpora. Single-target-language model (always outputs
English); no source-language prefix tag needed, confirmed from the
model's own usage examples. Apache-2.0, 0.2B params.
"""
import asyncio
import logging
from functools import lru_cache

from . import Translator

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load(repo, device):
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(repo)
    model = MarianMTModel.from_pretrained(repo).to(device).eval()
    return model, tokenizer


class HelsinkiIncTranslator(Translator):
    """Translates Indic-language text (including Sylheti) to English."""

    MODELS = ("inc-en",)
    DEFAULT_MODEL = "inc-en"
    REPOS = {"inc-en": "Helsinki-NLP/opus-mt-tc-bible-big-inc-en"}

    @classmethod
    def create(cls, model=None, device="cpu", **kwargs):
        return cls(model=model, device=device)

    def __init__(self, model=None, device="cpu"):
        self.model = model or self.DEFAULT_MODEL
        if self.model not in self.MODELS:
            raise ValueError(f"Unknown translation model {self.model!r}, "
                             f"expected one of {self.MODELS}")
        self.device = device

    def _translate(self, text):
        import torch
        model, tokenizer = _load(self.REPOS[self.model], self.device)
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=512)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    async def translate(self, text, source_language, language_name=None):
        try:
            return await asyncio.to_thread(self._translate, text)
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text