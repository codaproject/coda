"""Translation through a MADLAD-400 sequence-to-sequence model under CTranslate2."""
import asyncio
import logging
from functools import lru_cache

from . import Translator
from .ct2 import SENTENCE_END

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load(repo, device, compute_type):
    """Load a CTranslate2 MADLAD model with its SentencePiece tokenizer."""
    import ctranslate2
    import sentencepiece
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo)
    translator = ctranslate2.Translator(path, device=device,
                                        compute_type=compute_type)
    tokenizer = sentencepiece.SentencePieceProcessor(
        model_file=f"{path}/sentencepiece.model")
    return translator, tokenizer


class MadladTranslator(Translator):
    """Translates a transcript to English with a MADLAD-400 checkpoint.

    MADLAD-400 covers over 400 languages and infers the source language
    itself, so only the target is given. The weights are Apache-2.0.
    """

    MODELS = ("madlad400-3b-mt", "madlad400-7b-mt")
    DEFAULT_MODEL = "madlad400-3b-mt"
    # Int8 CTranslate2 conversions of google/madlad400-*
    REPOS = {
        "madlad400-3b-mt": "santhosh/madlad400-3b-ct2",
        "madlad400-7b-mt": "Heng666/madlad400-7b-mt-ct2-int8",
    }
    TARGET_TAG = "<2en>"

    @classmethod
    def create(cls, model=None, device="cpu", compute_type="int8", **kwargs):
        return cls(model=model, device=device, compute_type=compute_type)

    def __init__(self, model=None, device="cpu", compute_type="int8"):
        self.model = model or self.DEFAULT_MODEL
        if self.model not in self.MODELS:
            raise ValueError(f"Unknown translation model {self.model!r}, "
                             f"expected one of {self.MODELS}")
        # CTranslate2 has no MPS backend
        self.device = "cpu" if device == "mps" else device
        self.compute_type = compute_type

    def load(self):
        return _load(self.REPOS[self.model], self.device, self.compute_type)

    def _translate(self, text):
        translator, tokenizer = self.load()
        sentences = [s for s in SENTENCE_END.split(text.strip()) if s.strip()]
        if not sentences:
            return ""
        # SentencePiece leaves out the end-of-sequence token T5 expects, and
        # without it the model runs on past the sentence
        tokens = [tokenizer.encode(f"{self.TARGET_TAG} {s}", out_type=str)
                  + ["</s>"] for s in sentences]
        results = translator.translate_batch(tokens, max_decoding_length=512)
        return " ".join(tokenizer.decode(r.hypotheses[0]) for r in results)

    async def translate(self, text, source_language, language_name=None):
        try:
            return await asyncio.to_thread(self._translate, text)
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
