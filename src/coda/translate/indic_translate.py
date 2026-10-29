"""Translation through bodhan-ai/indic-translate (Gemma 4 E4B fine-tune).

Requires transformers>=5.12, torch>=2.6, pillow, torchvision (Gemma4's
processor pulls in image/audio sub-components even for text-only use).
Gated model -- needs HF_TOKEN with the license accepted on the Hub.

Weights are 15.9GB bf16; the model card's stated minimum is a 24GB GPU.
On 24GB unified-memory Apple Silicon, this offloads part of the model to
disk and runs at ~400s per short sentence -- confirmed too slow for this
pipeline on that hardware, kept here for hardware with more headroom
(the model card recommends 40GB+ for full-length documents).
"""
import asyncio
import logging
from functools import lru_cache
import os
from dotenv import load_dotenv

load_dotenv()

from . import Translator

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load(repo, device):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    token = os.environ.get("HF_TOKEN")
    processor = AutoProcessor.from_pretrained(repo, token=token)
    model = AutoModelForImageTextToText.from_pretrained(
        repo, dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager", token=token,
    ).eval()
    return model, processor


class IndicTranslateTranslator(Translator):
    """Translates a transcript to English with bodhan-ai/indic-translate.

    Source language is not specified in the prompt; the model infers it.
    Only English-as-target is wired here, matching every other backend
    in this project.
    """

    MODELS = ("indic-translate",)
    DEFAULT_MODEL = "indic-translate"
    REPOS = {"indic-translate": "bodhan-ai/indic-translate"}
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
        import torch
        model, processor = _load(self.REPOS[self.model], self.device)
        messages = [{
            "role": "user",
            "content": f"Translate the following text into {self.TARGET_LANGUAGE}:\n\n{text}",
        }]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=512, do_sample=False, use_cache=True)
        completion = output[0, inputs["input_ids"].shape[-1]:]
        return processor.decode(completion, skip_special_tokens=True).strip()

    async def translate(self, text, source_language, language_name=None):
        try:
            return await asyncio.to_thread(self._translate, text)
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text