"""Translation through a Tencent Hy-MT2 causal LM, prompted for the task.

Requires transformers>=5.6.0 and trust_remote_code=True.
"""
import asyncio
import logging
from functools import lru_cache

from . import Translator

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load(repo, device):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
    return model, tokenizer


class HyMT2Translator(Translator):
    """Translates a transcript to English with a Tencent Hy-MT2 checkpoint.

    Unlike ct2.py, this does not sentence-split before translating --
    unverified on transcripts longer than what's been tested. Uses
    sampling (not greedy decoding), so output is not deterministic run
    to run.
    """

    MODELS = ("hy-mt2-1.8b", "hy-mt2-7b")
    DEFAULT_MODEL = "hy-mt2-1.8b"
    REPOS = {
        "hy-mt2-1.8b": "tencent/Hy-MT2-1.8B",
        "hy-mt2-7b": "tencent/Hy-MT2-7B",
    }
    PROMPT_TEMPLATE = (
        "Translate the following text into {target_lang}. Note that you "
        "should only output the translated result without any additional "
        "explanation: {source_text}"
    )
    TARGET_LANGUAGE = "English"

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
        prompt = self.PROMPT_TEMPLATE.format(
            target_lang=self.TARGET_LANGUAGE, source_text=text)
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(
            inputs, max_new_tokens=512, temperature=0.7, top_p=0.6,
            top_k=20, repetition_penalty=1.05, do_sample=True)
        return tokenizer.decode(
            out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()

    async def translate(self, text, source_language, language_name=None):
        try:
            return await asyncio.to_thread(self._translate, text)
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text