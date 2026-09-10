"""CHAMPS embedding-based cause-of-death inference agent for CODA.

Embeds the accumulated dialogue transcript and scores it against one or more
scikit-learn classifiers (one per CHAMPS age group, plus a generic
age-agnostic fallback), predicting CHAMPS cause-of-death groups.
"""
import logging
from typing import Dict, List, Optional

import joblib
from gilda import Annotation

from coda.embedding import BaseEmbedder, SentenceTransformerEmbedder
from coda.inference.agent import InferenceAgent, InferenceServer
from coda.inference.champs_prompted_agent import CHAMPS_GROUP_TO_ICD10
from coda.metadata import Metadata

logger = logging.getLogger(__name__)

# Upper bound in days for each CHAMPS age-group model, checked in order.
AGE_GROUP_BOUNDARIES = [
    ("less24hours", 1),
    ("earlyneonate", 7),
    ("lateneonate", 28),
    ("infant", 365.25),
    ("child", 5 * 365.25),
]


def _age_group_key(metadata: Optional[Metadata]) -> str:
    """Map case metadata to a CHAMPS age-group model key, or "generic" if unknown."""
    profile = metadata.profile if metadata else None
    if profile is None:
        return "generic"
    if profile.stillbirth:
        return "stillbirth"
    age_days = profile.age.days if profile.age else None
    if age_days is None:
        return "generic"
    for key, upper_bound in AGE_GROUP_BOUNDARIES:
        if age_days < upper_bound:
            return key
    return "generic"


class EmbeddingCODAgent(InferenceAgent):
    """Cause-of-death inference agent driven by a dialogue-embedding classifier.

    Embeds the accumulated transcript each chunk and scores it against a
    scikit-learn classifier selected by age group, falling back to a generic
    model when no age-specific model is available.
    """

    def __init__(self, embedder: BaseEmbedder, models: Dict[str, object]):
        super().__init__()
        self.embedder = embedder
        self.models = models
        self.dialogue_embedding = None

    def reset(self):
        super().reset()
        self.dialogue_embedding = None

    def _select_model(self):
        key = _age_group_key(self.metadata)
        return self.models.get(key) or self.models.get("generic")

    async def infer(self, chunk_id: str, text: str,
                    annotations: List[Annotation]) -> dict:
        model = self._select_model()
        if model is None:
            return {
                "causes": {
                    "icd10:R99": {
                        "name": "Other ill-defined and unspecified causes of mortality",
                        "identifiers": {"icd10": "R99"},
                        "score": 1.0,
                    }
                },
                "reasoning": "No embedding classifier available.",
            }

        self.dialogue_embedding = self.embedder.embed(self.all_text.strip())
        probabilities = model.predict_proba([self.dialogue_embedding])[0]

        causes = {}
        for group_name, probability in zip(model.classes_, probabilities):
            icd10 = CHAMPS_GROUP_TO_ICD10.get(group_name)
            if icd10 is None:
                logger.warning("No ICD-10 mapping for CHAMPS group '%s'", group_name)
                continue
            causes[f"icd10:{icd10}"] = {
                "name": group_name,
                "identifiers": {"icd10": icd10},
                "score": float(probability),
            }

        return {
            "causes": causes,
            "reasoning": "Predicted from a dialogue-text embedding classifier.",
        }


def create_embedding_agent(model_path: str,
                            embed_model: str = "all-MiniLM-L6-v2") -> EmbeddingCODAgent:
    """Build an EmbeddingCODAgent from a joblib bundle of trained classifiers.

    `model_path` points at a bundle produced by champs_statsML's
    export_coda_dialogue_model.py: a dict with a "generic" scikit-learn
    classifier and a "by_age" dict of per-age-group classifiers, all trained
    on embeddings from `embed_model` (or `bundle["embed_model"]` if present).
    """
    bundle = joblib.load(model_path)
    models = {"generic": bundle["generic"], **bundle.get("by_age", {})}
    embedder = SentenceTransformerEmbedder(
        model_name=bundle.get("embed_model", embed_model))
    return EmbeddingCODAgent(embedder=embedder, models=models)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CODA embedding-based inference agent server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", required=True,
                        help="Path to a joblib bundle exported by champs_statsML")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from coda.config import settings

    agent = create_embedding_agent(args.model_path)
    server = InferenceServer(agent, host=settings.inference.host,
                             port=settings.inference.port)
    server.run()
