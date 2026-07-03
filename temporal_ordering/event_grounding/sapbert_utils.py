"""
Semantic (dense-retrieval) grounder over the SNOMED CT + HPO terms we add to GILDA.

This complements the exact-match GILDA grounder: phrases that fail to ground by
normalized string match can be sent here to retrieve the nearest concepts in an
embedding space.

Model
-----
We embed terms with `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`. SapBERT is a
plain BERT encoder (NOT a sentence-transformers model), so we load it through
`transformers` and pool manually. Following SapBERT's official usage, the
sentence/term embedding is the **[CLS] token** of the last hidden state
(`last_hidden_state[:, 0, :]`). We L2-normalize the vectors so that cosine
similarity == dot product, and store them in a ChromaDB collection configured
with cosine space.

Term set
--------
  - SNOMED CT terms from `parse_rf2_terms(data/snomed)` (disorder/procedure/finding)
  - HP (HPO) terms pulled from GILDA's default grounder
Every surface form (FSN + every synonym) is embedded as its own vector, with the
concept id / name / namespace kept as metadata, which is what SapBERT-style
entity linking wants for recall.

This module is a library. The ChromaDB vector database is constructed by the
companion `build_chromadb.py` script; here we provide the shared building blocks
(`SapBERTEncoder`, `collect_terms`, `_get_chroma_client`) and the query-time
interface.

Programmatic use (e.g. to back up the GILDA grounder on ungrounded phrases):
  from sapbert_utils import load_semantic_grounder
  sg = load_semantic_grounder()
  results = sg.ground(["difficulty breathing", "swollen belly"], top_k=5)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

from coda import CODA_BASE

import logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
SNOMED_DATA = Path(__file__).parent / "snomed_data"
CHROMA_PATH = CODA_BASE.module("temporal_ordering", "event_grounding").join("chroma_sapbert")
COLLECTION_NAME = "snomed_hp_sapbert"

# HPO namespaces to pull from GILDA's default grounder
DEFAULT_GROUNDER_DBS = ("HP",)

DEFAULT_BATCH_SIZE = 128
MAX_SEQ_LEN = 32  # SNOMED terms / phrases are short; keep it tight for speed.

# ChromaDB caps the number of records per add() call.
CHROMA_ADD_BATCH = 4000


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Embedding: SapBERT with [CLS] pooling
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Term collection 
# ---------------------------------------------------------------------------

@dataclass
class TermRecord:
    text: str          # surface form to embed
    db: str            # namespace, e.g. SNOMEDCT_disorder, HP
    concept_id: str    # concept identifier
    name: str          # entry/preferred name
    status: str        # "name" | "synonym"


def collect_terms(snomed_root: Path = SNOMED_DATA, hp_grounder=None) -> list[TermRecord]:
    """Collect the SNOMED + HP surface forms we ground with, deduplicated.

    `hp_grounder` supplies the HP terms via its `.entries`; when None, GILDA's
    default grounder is loaded. Injecting one lets callers (and tests) provide a
    custom term set instead of the full default database.
    """
    from gilda import Grounder

    from .snomed_rf2_utils import _parse_rf2_terms

    logger.info(f"  Parsing SNOMED RF2 release from {snomed_root} ...")
    snomed_terms = _parse_rf2_terms(snomed_root)

    wanted = set(DEFAULT_GROUNDER_DBS)
    logger.info(f"  Extracting {', '.join(DEFAULT_GROUNDER_DBS)} terms from GILDA's default grounder ...")
    if hp_grounder is None:
        hp_grounder = Grounder()
    hp_terms = [
        t
        for terms in hp_grounder.entries.values()
        for t in terms
        if t.db in wanted
    ]

    records: list[TermRecord] = []
    # Dedupe on (surface text, db, concept id) so we don't store identical rows.
    seen: set[tuple[str, str, str]] = set()
    added: Counter = Counter()
    for t in (*snomed_terms, *hp_terms):
        text = (t.text or "").strip()
        if not text:
            continue
        key = (text.lower(), t.db, t.id)
        if key in seen:
            continue
        seen.add(key)
        records.append(TermRecord(
            text=text,
            db=t.db,
            concept_id=t.id,
            name=t.entry_name,
            status=t.status,
        ))
        added[t.db] += 1

    logger.info(f"  Collected {len(records):,} unique surface forms across {len(added)} namespaces")
    return records


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def _get_chroma_client(path: Path = CHROMA_PATH):
    import chromadb

    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


# ---------------------------------------------------------------------------
# Query mode
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    identifier: str    # "<db>:<concept_id>"
    name: str
    db: str
    concept_id: str
    text: str          # matched surface form
    score: float       # cosine similarity in [0, 1]


class IndexQueryUtil:
    """Query interface over the prebuilt SapBERT vector database."""

    def __init__(self, chroma_path: Path = CHROMA_PATH, embedder: SapBERTEncoder | None = None):
        client = _get_chroma_client(chroma_path)
        try:
            self.collection = client.get_collection(COLLECTION_NAME)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"Could not open collection '{COLLECTION_NAME}' at {chroma_path}. "
                f"Build it first with `python semantic_grounder.py build`. ({exc})"
            )
        self.embedder = embedder or SapBERTEncoder()

    def ground(
        self,
        queries: List[str],
        top_k: int = 5,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[List[Candidate]]:
        """Return, for each query, the top_k nearest concept candidates (best first)."""
        
        if not queries:
            return []

        embs = self.embedder.encode(queries, batch_size=batch_size)
        res = self.collection.query(
            query_embeddings=embs.tolist(),
            n_results=top_k,
        )

        # ChromaDB types these fields as Optional (they're absent unless requested
        # via include=); default to empty lists so zip() gets real iterables.
        all_metadatas = res["metadatas"] or []
        all_documents = res["documents"] or []
        all_distances = res["distances"] or []

        results = []
        for metadatas, documents, distances in zip(
            all_metadatas, all_documents, all_distances
        ):
            cands = []
            for meta, doc, dist in zip(metadatas, documents, distances):
                db = str(meta["db"])
                concept_id = str(meta["concept_id"])
                cands.append(Candidate(
                    identifier=f"{db}:{concept_id}",
                    name=str(meta["name"]),
                    db=db,
                    concept_id=concept_id,
                    text=doc,
                    score=round(1.0 - dist, 4),  # cosine distance -> similarity
                ))
            results.append(cands)

        return results


def load_semantic_grounder(
    chroma_path: Path = CHROMA_PATH, embedder: SapBERTEncoder | None = None
) -> IndexQueryUtil:
    """Convenience constructor for programmatic use."""
    return IndexQueryUtil(chroma_path=chroma_path, embedder=embedder)
