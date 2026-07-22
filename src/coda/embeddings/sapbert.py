"""SapBERT encoder for short biomedical strings.

We embed terms with ``cambridgeltl/SapBERT-from-PubMedBERT-fulltext``. SapBERT is
a plain BERT encoder (NOT a sentence-transformers model), so we load it through
``transformers`` and pool manually. Following SapBERT's official usage, the
term embedding is the **[CLS] token** of the last hidden state
(``last_hidden_state[:, 0, :]``); we L2-normalize the vectors so cosine
similarity equals the dot product.

This module is framework-neutral (only ``torch`` + ``transformers``) so it can
back both the ChromaDB-based semantic grounder
(:mod:`coda.grounding.temporal_ordering.event_grounding.sapbert_utils`) and the
SNOMED CT knowledge-graph embeddings (:mod:`coda.kg.sources.snomedct_embeddings`).
"""
from __future__ import annotations

import logging

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

DEFAULT_BATCH_SIZE = 128
MAX_SEQ_LEN = 32  # SNOMED terms / phrases are short; keep it tight for speed.


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SapBERTEncoder:
    """Encode short biomedical strings into L2-normalized SapBERT [CLS] embeddings."""

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or _select_device()
        logger.info(f"  Loading {model_name} on device '{self.device}' ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Return a (len(texts), hidden) float32 tensor of normalized embeddings on CPU."""
        all_embs: list[torch.Tensor] = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding", unit="batch")

        for start in iterator:
            batch = texts[start:start + batch_size]
            toks = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**toks)
            # SapBERT: use the [CLS] token (first position) of the last hidden state.
            cls = out.last_hidden_state[:, 0, :]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            all_embs.append(cls.to("cpu", dtype=torch.float32))

        if not all_embs:
            return torch.empty((0, self.model.config.hidden_size), dtype=torch.float32)
        return torch.cat(all_embs, dim=0)
