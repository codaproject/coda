"""Tests for the KG-backed GILDA event grounder (:mod:`snomed_kg_utils`).

The grounder no longer parses RF2 from disk; it queries the CODA KG for
``snomedct`` surface-form nodes and builds :class:`gilda.Term` objects from them.

Two layers of coverage:

* **Unit tests** (always run) stub the Neo4j driver with a fake session that
  returns canned node records, so they exercise Term construction and grounding
  end-to-end without a running Neo4j. This is the default/CI behavior.
* **Live integration test** (``test_make_gilda_grounder_live_kg``) runs only when
  a real Neo4j is reachable at ``CODA_KG_URL``; it seeds a few uniquely-keyed
  ``snomedct`` nodes, builds the real grounder against them, and tears down
  exactly those nodes. When no KG is detected it is skipped and the fake-driver
  unit tests remain the sole behavior.
"""
from __future__ import annotations

import pytest

from coda.grounding.temporal_ordering.event_grounding import snomed_kg_utils
from coda.grounding.temporal_ordering.event_grounding.snomed_kg_utils import (
    SOURCE,
    make_gilda_grounder,
    snomedct_terms_available,
)
from coda.runtime_config import get_kg_url

ASTHMA = "100000001"
NEPHROSTOMY = "100000002"
FEVER = "100000003"

# One record per surface form, mirroring what coda.kg.sources.snomedct writes:
# an FSN node (status="name") per concept plus synonym nodes (status="synonym").
# Synonym nodes carry their concept's semantic-tag category too (the exporter
# propagates the concept category to every surface form).
_NODES = [
    {"text": "Asthma (disorder)", "category": "disorder", "code": ASTHMA,
     "name": "Asthma (disorder)", "status": "name"},
    {"text": "Asthma", "category": "disorder", "code": ASTHMA,
     "name": "Asthma (disorder)", "status": "synonym"},
    {"text": "Bronchial asthma", "category": "disorder", "code": ASTHMA,
     "name": "Asthma (disorder)", "status": "synonym"},
    {"text": "Nephrostomy (procedure)", "category": "procedure", "code": NEPHROSTOMY,
     "name": "Nephrostomy (procedure)", "status": "name"},
    {"text": "Fever (finding)", "category": "finding", "code": FEVER,
     "name": "Fever (finding)", "status": "name"},
]


class _FakeResult:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def single(self):
        return self._records[0] if self._records else None


class _FakeSession:
    def __init__(self, nodes, count_present):
        self._nodes = nodes
        self._count_present = count_present

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **_params):
        if "count(n)" in query:
            return _FakeResult([{"present": self._count_present}])
        return _FakeResult(list(self._nodes))


class _FakeDriver:
    def __init__(self, nodes, count_present):
        self._nodes = nodes
        self._count_present = count_present
        self.closed = False

    def session(self):
        return _FakeSession(self._nodes, self._count_present)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_kg(monkeypatch):
    """Patch GraphDatabase.driver in snomed_kg_utils to return a fake driver."""
    created: list[_FakeDriver] = []

    def _make(nodes=_NODES, count_present=True):
        def _driver(url, auth=None):
            drv = _FakeDriver(nodes, count_present)
            created.append(drv)
            return drv

        monkeypatch.setattr(snomed_kg_utils.GraphDatabase, "driver", _driver)
        return created

    return _make


def test_make_gilda_grounder_grounds_queries(fake_kg):
    created = fake_kg()
    grounder = make_gilda_grounder()

    matches = grounder.ground("asthma")
    assert matches, "expected 'asthma' to ground to a SNOMED concept"
    assert matches[0].term.id == ASTHMA
    assert matches[0].term.db == "SNOMEDCT_disorder"
    assert matches[0].term.source == SOURCE

    # A registered synonym grounds back to its concept id.
    syn_matches = grounder.ground("bronchial asthma")
    assert syn_matches
    assert syn_matches[0].term.id == ASTHMA

    # Category drives the db namespace, mirroring the old RF2-derived shape.
    fever = grounder.ground("fever (finding)")
    assert fever and fever[0].term.db == "SNOMEDCT_finding"

    # A term never loaded from the KG does not ground.
    assert not grounder.ground("caffeine")

    # The driver opened for loading is closed afterwards.
    assert created and created[-1].closed


def test_snomedct_terms_available_true(fake_kg):
    fake_kg(count_present=True)
    assert snomedct_terms_available() is True


def test_snomedct_terms_available_false_when_empty(fake_kg):
    fake_kg(nodes=[], count_present=False)
    assert snomedct_terms_available() is False


# =========================================================================== #
# Live integration: run against a real Neo4j when one is reachable, else skip.
# =========================================================================== #

# Unique, obviously-synthetic ids/code so this test never collides with or
# deletes real SNOMED CT data in a shared KG. Teardown removes only these ids.
_TEST_CONCEPT_ID = "pytest-999000001"
_TEST_CONCEPT_CURIE = f"snomedct:{_TEST_CONCEPT_ID}"
_TEST_SYNONYM_CURIE = "snomedct:pytest-999000001-syn"
_TEST_IDS = [_TEST_CONCEPT_CURIE, _TEST_SYNONYM_CURIE]

# Surface forms distinctive enough to ground unambiguously even when the KG
# already holds the full real SNOMED CT release alongside these test nodes.
_TEST_FSN_TEXT = "Pytest synthetic asthma syndrome (disorder)"
_TEST_SYNONYM_TEXT = "Pytest synthetic wheezy affliction"


def _kg_reachable() -> bool:
    """True iff a Neo4j server answers a trivial query at CODA_KG_URL."""
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(get_kg_url(), auth=None)
    except Exception:  # noqa: BLE001 -- any driver/config error means "not reachable"
        return False
    try:
        with driver.session() as session:
            session.run("RETURN 1").single()
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        driver.close()


_KG_LIVE = _kg_reachable()


@pytest.fixture
def live_snomed_nodes():
    """Seed a few uniquely-keyed ``snomedct`` nodes and remove them afterwards.

    Only the exact seeded ids are deleted (never a label-wide delete), so this is
    safe to run against a KG that already contains real SNOMED CT data.
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(get_kg_url(), auth=None)
    nodes = [
        {"id": _TEST_CONCEPT_CURIE, "text": _TEST_FSN_TEXT, "name": _TEST_FSN_TEXT,
         "category": "disorder", "code": _TEST_CONCEPT_ID, "status": "name"},
        {"id": _TEST_SYNONYM_CURIE, "text": _TEST_SYNONYM_TEXT, "name": _TEST_FSN_TEXT,
         "category": "disorder", "code": _TEST_CONCEPT_ID, "status": "synonym"},
    ]
    try:
        with driver.session() as session:
            session.run(
                """
                UNWIND $nodes AS r
                MERGE (n:snomedct {id: r.id})
                SET n.text = r.text, n.name = r.name, n.category = r.category,
                    n.code = r.code, n.status = r.status
                """,
                nodes=nodes,
            )
        yield
    finally:
        with driver.session() as session:
            session.run(
                "MATCH (n:snomedct) WHERE n.id IN $ids DETACH DELETE n",
                ids=_TEST_IDS,
            )
        driver.close()


@pytest.mark.skipif(
    not _KG_LIVE,
    reason="No Neo4j reachable at CODA_KG_URL; using the fake-driver unit tests instead.",
)
def test_make_gilda_grounder_live_kg(live_snomed_nodes):
    # The KG is reachable and now holds our seeded snomedct nodes.
    assert snomedct_terms_available() is True

    grounder = make_gilda_grounder()

    # The seeded FSN grounds to the synthetic concept id with the expected db.
    # GILDA matches on normalized text, and the FSN carries its "(disorder)"
    # semantic tag verbatim, so the query includes it too.
    fsn = grounder.ground("pytest synthetic asthma syndrome (disorder)")
    assert fsn, "expected the seeded FSN to ground"
    assert fsn[0].term.id == _TEST_CONCEPT_ID
    assert fsn[0].term.db == "SNOMEDCT_disorder"
    assert fsn[0].term.source == SOURCE

    # Its synonym grounds back to the same concept id.
    syn = grounder.ground("pytest synthetic wheezy affliction")
    assert syn
    assert syn[0].term.id == _TEST_CONCEPT_ID
