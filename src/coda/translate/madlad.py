"""Translation through a MADLAD-400 sequence-to-sequence model."""
import asyncio
import logging
from functools import lru_cache

from . import Translator

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load(repo, device):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        repo, dtype=torch.bfloat16).to(device).eval()
    return model, tokenizer


class MadladTranslator(Translator):
    """Translates a transcript to English with a MADLAD-400 checkpoint.

    Unlike ct2.py, this does not sentence-split before translating --
    unverified on transcripts longer than what's been tested.
    """

    MODELS = ("madlad400-3b-mt", "madlad400-7b-mt", "madlad400-7b-mt-bt")
    DEFAULT_MODEL = "madlad400-3b-mt"
    REPOS = {
        "madlad400-3b-mt": "google/madlad400-3b-mt",
        "madlad400-7b-mt": "google/madlad400-7b-mt",
        "madlad400-7b-mt-bt": "google/madlad400-7b-mt-bt",
    }
    TARGET_TAG = "<2en>"

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
        model, tokenizer = _load(self.REPOS[self.model], self.device)
        inputs = tokenizer(f"{self.TARGET_TAG} {text}", return_tensors="pt").to(self.device)
        ids = model.generate(**inputs, max_new_tokens=512)
        return tokenizer.decode(ids[0], skip_special_tokens=True)

    async def translate(self, text, source_language, language_name=None):
        try:
            return await asyncio.to_thread(self._translate, text)
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text