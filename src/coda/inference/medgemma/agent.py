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
from coda.inference.agent import InferenceAgent, InferenceServer
from coda.inference.medgemma.engine import MedGemmaModel
from coda.inference.medgemma.prompt import MenuPrompt, build_menu_prompt
from coda.resources import get_resource_path

logger = logging.getLogger(__name__)


def load_menu(resource: str) -> dict[str, str]:
    """Load the code-to-description menu of allowed causes."""
    with open(get_resource_path(os.path.join("champs", resource))) as fh:
        return json.load(fh)


def medgemma_config():
    return getattr(settings.inference, "medgemma", None)


class MedGemmaChampsInferenceAgent(InferenceAgent):
    """Cause-of-death agent backed by the fine-tuned MedGemma CHAMPS model."""

    def __init__(
        self,
        menu: dict[str, str],
        adapter_path: str,
        *,
        model: Optional[MedGemmaModel] = None,
        top_k: int = 3,
        batch_size: int = 8,
    ):
        super().__init__()
        self.menu = menu
        self.menu_codes = list(menu.keys())
        self.adapter_path = os.path.expanduser(adapter_path)
        self.top_k = top_k
        self.batch_size = batch_size
        self._model = model

    def ensure_model(self) -> MedGemmaModel:
        if self._model is None:
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

    async def infer(self, chunk_id: str, text: str,
                    annotations: List[Annotation]) -> dict:
        narrative = self.all_text.strip()
        if not narrative:
            return {"causes": {}, "reasoning": "No narrative text yet."}

        try:
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


def create_medgemma_agent(
    adapter_path: Optional[str] = None,
    menu_resource: Optional[str] = None,
    **kwargs,
) -> MedGemmaChampsInferenceAgent:
    cfg = medgemma_config()
    adapter_path = adapter_path or (cfg and cfg.adapter_path)
    if not adapter_path:
        raise ValueError(
            "No MedGemma adapter path configured "
            "(set inference.medgemma.adapter_path)."
        )
    menu_resource = menu_resource or (cfg and cfg.menu) or "icd_all_labels.json"
    top_k = int((cfg and cfg.get("top_k")) or 3)
    batch_size = int((cfg and cfg.get("batch_size")) or 8)

    return MedGemmaChampsInferenceAgent(
        menu=load_menu(menu_resource),
        adapter_path=adapter_path,
        top_k=top_k,
        batch_size=batch_size,
        **kwargs,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    agent = create_medgemma_agent()
    agent.ensure_model()
    server = InferenceServer(
        agent,
        host=settings.inference.host,
        port=settings.inference.port,
    )
    server.run()
