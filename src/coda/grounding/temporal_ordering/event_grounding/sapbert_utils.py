"""
Semantic (dense-retrieval) grounder over the SNOMED CT terms in the KG.

This complements the exact-match GILDA grounder: phrases that fail to ground by
normalized string match can be sent here to retrieve the nearest concepts in an
embedding space.

Model
-----
Queries are embedded with `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`.
SapBERT is a plain BERT encoder (NOT a sentence-transformers model), so we load
it through `transformers` and pool manually. Following SapBERT's official usage,
the sentence/term embedding is the **[CLS] token** of the last hidden state
(`last_hidden_state[:, 0, :]`), L2-normalized so cosine similarity == dot
product.

Vector store
------------
The concept vectors live in the KG: the SNOMED CT source
(`coda.kg.sources.snomedct`) embeds every surface form (FSN + synonyms) with the
same SapBERT model and imports them as `snomedct` nodes, behind the Neo4j vector
index `snomedct_embedding` (created by `coda.kg.vector_index`). This module only
provides the query-time interface; it does not build the index.

Only SNOMED CT is served here — HPO terms are not embedded in the KG's SapBERT
space (see `coda.grounding.rag_grounder.retriever` for the same distinction).

Programmatic use (e.g. to back up the GILDA grounder on ungrounded phrases):
  from sapbert_utils import load_semantic_grounder
  sg = load_semantic_grounder()
  results = sg.ground(["difficulty breathing", "swollen belly"], top_k=5)
"""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import GraphDatabase

# SapBERTEncoder and its constants now live in a framework-neutral module so the
# KG exporter can reuse them; re-exported here to preserve this module's API.
from coda.embeddings.sapbert import (  # noqa: F401
    DEFAULT_BATCH_SIZE,
    MODEL_NAME,
    SapBERTEncoder,
)
from coda.runtime_config import get_kg_url

import logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Neo4j vector index over the SNOMED CT surface-form nodes. Created at KG
# startup as `{label}_{property}` = `snomedct_embedding` (see
# coda.kg.vector_index). SNOMED nodes carry a `snomedct:<id>` concept curie, so
# a synonym hit resolves to its concept without a graph hop.
DEFAULT_INDEX_NAME = "snomedct_embedding"

# Namespace reported for every candidate; matches the KG node label.
SNOMED_DB = "snomedct"


# ---------------------------------------------------------------------------
# Query mode
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    identifier: str    # concept curie, e.g. "snomedct:<concept_id>"
    name: str
    db: str
    concept_id: str
    text: str          # matched surface form
    score: float       # cosine similarity in [0, 1]


class IndexQueryUtil:
    """Query interface over the KG's SapBERT vector index."""

    def __init__(
        self,
        index_name: str = DEFAULT_INDEX_NAME,
        embedder: SapBERTEncoder | None = None,
        kg_url: str | None = None,
    ):
        self.index_name = index_name
        self.driver = GraphDatabase.driver(kg_url or get_kg_url(), auth=None)
        self.embedder = embedder or SapBERTEncoder()

    def close(self) -> None:
        self.driver.close()

    def ground(
        self,
        queries: list[str],
        top_k: int = 5,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[list[Candidate]]:
        """Return, for each query, the top_k nearest concept candidates (best first)."""

        if not queries:
            return []

        embs = self.embedder.encode(queries, batch_size=batch_size)

        results = []
        with self.driver.session() as session:
            # Neo4j's vector query takes a single embedding per call, so we issue
            # one query per input phrase.
            for emb in embs:
                records = session.run(
                    """
                    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                    YIELD node, score
                    RETURN node.concept_id AS concept_id, node.name AS name,
                           node.text AS text, node.code AS code, score
                    """,
                    index_name=self.index_name,
                    top_k=top_k,
                    embedding=emb.tolist(),
                )
                cands = [
                    Candidate(
                        identifier=str(record["concept_id"]),
                        name=str(record["name"]),
                        db=SNOMED_DB,
                        concept_id=str(record["code"]),
                        text=str(record["text"]),
                        # Neo4j returns cosine similarity directly in [0, 1].
                        score=round(float(record["score"]), 4),
                    )
                    for record in records
                ]
                results.append(cands)

        return results


def load_semantic_grounder(
    index_name: str = DEFAULT_INDEX_NAME,
    embedder: SapBERTEncoder | None = None,
    kg_url: str | None = None,
) -> IndexQueryUtil:
    """Convenience constructor for programmatic use."""
    return IndexQueryUtil(index_name=index_name, embedder=embedder, kg_url=kg_url)
