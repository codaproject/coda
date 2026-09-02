"""MedGemma CHAMPS cause-of-death inference agent for CODA.

Runs the fine-tuned MedGemma classifier over the verbal-autopsy narrative
accumulated during an interview, scoring a fixed menu of terminal ICD-11 causes
and returning the most likely ones as CODA cause objects.
"""

import asyncio
import json
import logging
import os
from typing import List, Optional

import torch
from gilda import Annotation

from coda.config import settings
from coda.inference.agent import InferenceAgent
from coda.inference.champs_finetuned.prompt import MenuPrompt, build_menu_prompt
from coda.resources import get_resource_path

logger = logging.getLogger(__name__)


def load_menu(resource: str) -> dict[str, str]:
    """Load the code-to-description menu of allowed causes."""
    with open(get_resource_path(os.path.join("champs", resource))) as fh:
        return json.load(fh)


def finetuned_config():
    return getattr(settings.inference, "champs_finetuned", None)


class ChampsFinetunedInferenceAgent(InferenceAgent):
    """Cause-of-death agent backed by the fine-tuned MedGemma CHAMPS model."""

    def __init__(
        self,
        menu: dict[str, str],
        adapter_path: str,
        *,
        model=None,
        top_k: int = 3,
        batch_size: int = 8,
        llm_semaphore: asyncio.Semaphore | None = None,
    ):
        super().__init__()
        self.menu = menu
        self.menu_codes = list(menu.keys())
        self.adapter_path = os.path.expanduser(adapter_path)
        self.top_k = top_k
        self.batch_size = batch_size
        self._model = model
        # Serialize access to the single shared model across sessions
        self.llm_semaphore = llm_semaphore or asyncio.Semaphore(1)

    def ensure_model(self):
        if self._model is None:
            from coda.inference.champs_finetuned.engine import MedGemmaModel
            logger.info("Loading MedGemma model with adapter %s", self.adapter_path)
            self._model = MedGemmaModel(adapter_path=self.adapter_path)
            logger.info("MedGemma model loaded")
        return self._model

    def score_menu(self, narrative: str) -> list[float]:
        model = self.ensure_model()
        prompt = build_menu_prompt(
            self.menu_codes, self.menu, narrative=narrative
        )
        candidates = [
            MenuPrompt.format_answer(i) for i in range(1, len(self.menu_codes) + 1)
        ]
        return model.score_candidates(
            prompt.messages, candidates, batch_size=self.batch_size
        )

    def create_session_agent(self) -> "ChampsFinetunedInferenceAgent":
        return ChampsFinetunedInferenceAgent(
            menu=self.menu,
            adapter_path=self.adapter_path,
            model=self._model,
            top_k=self.top_k,
            batch_size=self.batch_size,
            llm_semaphore=self.llm_semaphore,
        )

    async def infer(self, chunk_id: str, text: str,
                    annotations: List[Annotation]) -> dict:
        narrative = self.all_text.strip()
        if not narrative:
            return {"causes": {}, "reasoning": "No narrative text yet."}

        try:
            async with self.llm_semaphore:
                scores = await asyncio.to_thread(self.score_menu, narrative)
        except Exception:
            logger.exception("MedGemma inference failed for chunk %s", chunk_id)
            return {"causes": {}, "reasoning": "MedGemma inference raised an exception."}

        probs = torch.tensor(scores).softmax(0).tolist()
        ranked = sorted(
            zip(self.menu_codes, probs), key=lambda x: x[1], reverse=True
        )[: self.top_k]

        causes = {
            f"icd11:{code}": {
                "name": self.menu[code],
                "identifiers": {"icd11": code},
                "score": round(prob, 6),
            }
            for code, prob in ranked
        }

        top_code, top_prob = ranked[0]
        reasoning = (
            f"MedGemma top cause: {self.menu[top_code]} "
            f"({top_code}, p={top_prob:.2f}) from the verbal autopsy narrative."
        )
        return {"causes": causes, "reasoning": reasoning}


def create_champs_finetuned_agent(
    adapter_path: Optional[str] = None,
    menu_resource: Optional[str] = None,
    **kwargs,
) -> ChampsFinetunedInferenceAgent:
    cfg = finetuned_config()
    adapter_path = adapter_path or (cfg and cfg.adapter_path)
    if not adapter_path:
        raise ValueError(
            "No MedGemma adapter path configured "
            "(set inference.champs_finetuned.adapter_path)."
        )
    menu_resource = menu_resource or (cfg and cfg.menu) or "icd_all_labels.json"
    top_k = int((cfg and cfg.get("top_k")) or 3)
    batch_size = int((cfg and cfg.get("batch_size")) or 8)
    max_concurrency = int(settings.inference.get("max_concurrency", 1) or 1)
    kwargs.setdefault("llm_semaphore", asyncio.Semaphore(max_concurrency))

    return ChampsFinetunedInferenceAgent(
        menu=load_menu(menu_resource),
        adapter_path=adapter_path,
        top_k=top_k,
        batch_size=batch_size,
        **kwargs,
    )
