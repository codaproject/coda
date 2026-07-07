"""Integration tests for the SapBERT KG vector-query interface.

Concept vectors now live in the knowledge graph: the SNOMED CT source
(``coda.kg.sources.snomedct``) embeds every surface form with SapBERT and imports
them as ``snomedct`` nodes behind the Neo4j vector index ``snomedct_embedding``
(see ``coda.kg.vector_index``). ``IndexQueryUtil`` embeds a query with the same
model and retrieves the nearest concepts through that index.

Because the store is the KG, these are integration tests: they require a running
Neo4j reachable at ``CODA_KG_URL`` with the ``snomedct_embedding`` index
populated. When the KG is unreachable or the index is missing/empty, the whole
module is skipped rather than failed.

The real ``SapBERTEncoder`` does the embedding. It is loaded once per module and
injected via the ``encoder`` fixture, so the transformer weights load a single
time across the suite.
"""
from __future__ import annotations

import pytest

from coda.grounding.temporal_ordering.event_grounding.sapbert_utils import (
    DEFAULT_INDEX_NAME,
    SNOMED_DB,
    Candidate,
    IndexQueryUtil,
    SapBERTEncoder,
    load_semantic_grounder,
)
from coda.runtime_config import get_kg_url


# --------------------------------------------------------------------------- #
# KG availability: skip the whole module unless a populated index is reachable.
# --------------------------------------------------------------------------- #
def _kg_index_ready() -> bool:
    """True iff Neo4j is reachable and the SapBERT vector index has data."""
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(get_kg_url(), auth=None)
    except Exception:  # noqa: BLE001 -- any driver/config error means "not ready"
        return False
    try:
        with driver.session() as session:
            record = session.run(
                """
                SHOW VECTOR INDEXES YIELD name
                WHERE name = $index_name
                RETURN count(*) AS n
                """,
                index_name=DEFAULT_INDEX_NAME,
            ).single()
            if record is None or record["n"] == 0:
                return False
            # Require at least one embedded SNOMED node, else queries return nothing.
            populated = session.run(
                "MATCH (n:snomedct) WHERE n.embedding IS NOT NULL RETURN n LIMIT 1"
            ).single()
            return populated is not None
    except Exception:  # noqa: BLE001
        return False
    finally:
        driver.close()


pytestmark = pytest.mark.skipif(
    not _kg_index_ready(),
    reason=(
        "KG not reachable at CODA_KG_URL or the 'snomedct_embedding' index is "
        "missing/empty; skipping SapBERT vector-query integration tests."
    ),
)


@pytest.fixture(scope="module")
def encoder() -> SapBERTEncoder:
    """The real SapBERT encoder, loaded once and shared across the module."""
    return SapBERTEncoder()


@pytest.fixture
def util(encoder) -> IndexQueryUtil:
    query_util = IndexQueryUtil(embedder=encoder)
    yield query_util
    query_util.close()


# =========================================================================== #
# Query: IndexQueryUtil / load_semantic_grounder
# =========================================================================== #
def test_ground_returns_candidates_best_first(util):
    results = util.ground(["asthma", "nephrostomy"], top_k=3)

    assert len(results) == 2  # one candidate list per query
    for cands in results:
        assert len(cands) <= 3
        assert all(isinstance(c, Candidate) for c in cands)
        # Scores are cosine similarities in [0, 1], sorted best-first.
        scores = [c.score for c in cands]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)


def test_ground_candidate_shape(util):
    top = util.ground(["asthma"], top_k=1)[0][0]

    # Every SNOMED candidate reports the KG label as its namespace, and the
    # identifier is the concept curie, of which `code` is the bare id.
    assert top.db == SNOMED_DB
    assert top.identifier.startswith("snomedct:")
    assert top.identifier == f"snomedct:{top.concept_id}"
    assert top.name
    assert top.text


def test_ground_empty_queries_returns_empty(util):
    assert util.ground([]) == []


def test_ground_respects_top_k(util):
    (cands,) = util.ground(["fever"], top_k=2)
    assert len(cands) <= 2


def test_load_semantic_grounder_builds_query_util(encoder):
    grounder = load_semantic_grounder(embedder=encoder)
    try:
        assert isinstance(grounder, IndexQueryUtil)
        (cands,) = grounder.ground(["fever"], top_k=1)
        assert all(c.db == SNOMED_DB for c in cands)
    finally:
        grounder.close()
