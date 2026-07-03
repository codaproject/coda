"""Tests for the SapBERT vector-database build + query pipeline.

The pipeline in ``temporal_ordering.event_grounding`` has three moving parts:

  * ``collect_terms``      -- gather SNOMED + HP surface forms into ``TermRecord``s
  * ``build_database``     -- embed the unique strings and index them into ChromaDB
  * ``IndexQueryUtil``     -- embed a query and retrieve nearest concepts

Only the *dataset* is mocked -- both terminologies are supplied as small,
in-memory mock releases and fed through the real code path (no monkeypatching,
no fake embeddings):

  * **SNOMED** -- a tiny synthetic RF2 release tree is written under ``tmp_path``
    and parsed by the real ``_parse_rf2_terms`` (same approach as
    ``test_snomed_rf2_parser``), so no multi-gigabyte real release is needed.
  * **HPO**    -- a real ``gilda.Grounder`` built from a handful of mock ``HP``
    (and one non-HP) ``Term``s, injected via ``collect_terms``/``build_database``'s
    ``hp_grounder`` parameter, standing in for GILDA's full default grounder.

The real ``SapBERTEncoder`` does the embedding. It is loaded once per module and
injected everywhere via the ``encoder`` fixture, so the transformer weights load
a single time across the suite.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gilda import Grounder, Term

from temporal_ordering.event_grounding import build_chromadb
from temporal_ordering.event_grounding.sapbert_utils import (
    COLLECTION_NAME,
    Candidate,
    IndexQueryUtil,
    SapBERTEncoder,
    TermRecord,
    collect_terms,
    load_semantic_grounder,
)


@pytest.fixture(scope="module")
def encoder() -> SapBERTEncoder:
    """The real SapBERT encoder, loaded once and shared across the module."""
    return SapBERTEncoder()


# --------------------------------------------------------------------------- #
# Mock SNOMED: a minimal synthetic RF2 release tree
# --------------------------------------------------------------------------- #
MODULE_ID = "900000000000207008"
CASE_ID = "900000000000448009"
DEF_STATUS_ID = "900000000000074008"
EFFECTIVE = "20260601"

FSN_TYPE_ID = "900000000000003001"
SYNONYM_TYPE_ID = "900000000000013009"
US_ENGLISH_REFSET_ID = "900000000000509007"
PREFERRED_ID = "900000000000548007"

CONCEPT_HEADER = ["id", "effectiveTime", "active", "moduleId", "definitionStatusId"]
DESCRIPTION_HEADER = [
    "id", "effectiveTime", "active", "moduleId", "conceptId",
    "languageCode", "typeId", "term", "caseSignificanceId",
]
LANGUAGE_HEADER = [
    "id", "effectiveTime", "active", "moduleId",
    "refsetId", "referencedComponentId", "acceptabilityId",
]

ASTHMA = "100000001"       # disorder, FSN + one preferred synonym
NEPHROSTOMY = "100000002"  # procedure, FSN only
FEVER = "100000003"        # finding, FSN only

# Text of the surface forms the parser is expected to emit for the fixture.
ASTHMA_FSN = "Asthma (disorder)"
ASTHMA_SYN = "Bronchial asthma"
NEPHROSTOMY_FSN = "Nephrostomy (procedure)"
FEVER_FSN = "Fever (finding)"


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(header)] + ["\t".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _concept(cid: str, active: str = "1") -> list[str]:
    return [cid, EFFECTIVE, active, MODULE_ID, DEF_STATUS_ID]


def _description(desc_id, concept_id, type_id, term, active="1") -> list[str]:
    return [desc_id, EFFECTIVE, active, MODULE_ID, concept_id, "en", type_id, term, CASE_ID]


def _language(member_id, desc_id, acc=PREFERRED_ID, refset=US_ENGLISH_REFSET_ID) -> list[str]:
    return [member_id, EFFECTIVE, "1", MODULE_ID, refset, desc_id, acc]


def _write_snomed_tree(root: Path, concepts, descriptions, languages) -> Path:
    """Write an RF2 Snapshot tree (headers always present, rows as given)."""
    terminology = root / "Snapshot" / "Terminology"
    language = root / "Snapshot" / "Refset" / "Language"
    _write_tsv(terminology / f"sct2_Concept_Snapshot_INT_{EFFECTIVE}.txt", CONCEPT_HEADER, concepts)
    _write_tsv(
        terminology / f"sct2_Description_Snapshot-en_INT_{EFFECTIVE}.txt",
        DESCRIPTION_HEADER, descriptions,
    )
    _write_tsv(
        language / f"der2_cRefset_LanguageSnapshot-en_INT_{EFFECTIVE}.txt",
        LANGUAGE_HEADER, languages,
    )
    return root


@pytest.fixture
def snomed_root(tmp_path: Path) -> Path:
    """A three-concept synthetic RF2 release tree."""
    return _write_snomed_tree(
        tmp_path / "snomed",
        concepts=[_concept(ASTHMA), _concept(NEPHROSTOMY), _concept(FEVER)],
        descriptions=[
            _description("200000001", ASTHMA, FSN_TYPE_ID, ASTHMA_FSN),
            _description("200000002", ASTHMA, SYNONYM_TYPE_ID, ASTHMA_SYN),
            _description("200000010", NEPHROSTOMY, FSN_TYPE_ID, NEPHROSTOMY_FSN),
            _description("200000020", FEVER, FSN_TYPE_ID, FEVER_FSN),
        ],
        languages=[_language("300000002", "200000002")],  # accept the asthma synonym
    )


@pytest.fixture
def empty_snomed_root(tmp_path: Path) -> Path:
    """A valid-but-empty RF2 tree (headers only) -> parser yields no terms."""
    return _write_snomed_tree(tmp_path / "empty_snomed", concepts=[], descriptions=[], languages=[])


# --------------------------------------------------------------------------- #
# Mock HPO: a real gilda.Grounder built from mock HP (and one non-HP) terms
# --------------------------------------------------------------------------- #
HP_DIFFICULTY_BREATHING = ("HP:0002098", "Respiratory distress", "difficulty breathing")
HP_ABDOMINAL_SWELLING = ("HP:0003270", "Abdominal distention", "swollen belly")


def _hp_term(concept_id: str, name: str, text: str, status: str = "synonym") -> Term:
    return Term(
        norm_text=text.lower(),
        text=text,
        db="HP",
        id=concept_id,
        entry_name=name,
        status=status,
        source="HPO",
    )


@pytest.fixture
def hp_grounder() -> Grounder:
    """A real GILDA grounder over a mock HP term set (plus one non-HP term)."""
    terms = [
        _hp_term(*HP_DIFFICULTY_BREATHING, status="name"),
        # A duplicate surface form on the same concept -> must be deduped.
        _hp_term(*HP_DIFFICULTY_BREATHING, status="synonym"),
        _hp_term(*HP_ABDOMINAL_SWELLING),
        # A non-HP term that must be filtered out (only HP is wanted).
        Term(
            norm_text="aspirin", text="aspirin", db="MESH",
            id="D001241", entry_name="Aspirin", status="name", source="MESH",
        ),
    ]
    return Grounder(terms=terms)


@pytest.fixture
def built_db(snomed_root, hp_grounder, encoder, tmp_path) -> Path:
    """Build a ChromaDB index over the mock terms with the real encoder."""
    chroma_path = tmp_path / "chroma"
    build_chromadb.build_database(
        snomed_root=snomed_root,
        chroma_path=chroma_path,
        encoder=encoder,
        hp_grounder=hp_grounder,
    )
    return chroma_path


# =========================================================================== #
# collect_terms
# =========================================================================== #
def test_collect_terms_gathers_snomed_and_hp(snomed_root, hp_grounder):
    records = collect_terms(snomed_root, hp_grounder=hp_grounder)

    assert all(isinstance(r, TermRecord) for r in records)
    by_text = {r.text: r for r in records}

    # SNOMED surface forms (FSN + accepted synonym) are present with parsed metadata.
    assert by_text[ASTHMA_FSN].db == "SNOMEDCT_disorder"
    assert by_text[ASTHMA_FSN].concept_id == ASTHMA
    assert by_text[ASTHMA_FSN].status == "name"
    assert by_text[ASTHMA_SYN].status == "synonym"
    assert by_text[ASTHMA_SYN].concept_id == ASTHMA
    assert by_text[NEPHROSTOMY_FSN].db == "SNOMEDCT_procedure"
    assert by_text[FEVER_FSN].db == "SNOMEDCT_finding"

    # HP terms are pulled in with db == "HP".
    assert by_text["difficulty breathing"].db == "HP"
    assert by_text["difficulty breathing"].concept_id == "HP:0002098"
    assert by_text["swollen belly"].concept_id == "HP:0003270"


def test_collect_terms_filters_non_hp(snomed_root, hp_grounder):
    texts = {r.text for r in collect_terms(snomed_root, hp_grounder=hp_grounder)}

    # Only HP is wanted from the grounder; the MESH term is dropped.
    assert "aspirin" not in texts


def test_collect_terms_dedupes_identical_surface_forms(snomed_root, hp_grounder):
    records = collect_terms(snomed_root, hp_grounder=hp_grounder)

    # "difficulty breathing" appears twice on HP:0002098 (name + synonym) but must
    # collapse to a single record keyed on (text.lower, db, id).
    matches = [r for r in records if r.text == "difficulty breathing"]
    assert len(matches) == 1

    keys = [(r.text.lower(), r.db, r.concept_id) for r in records]
    assert len(keys) == len(set(keys))


# =========================================================================== #
# build_database
# =========================================================================== #
def test_build_database_indexes_all_records(built_db, snomed_root, hp_grounder):
    import chromadb

    expected = len(collect_terms(snomed_root, hp_grounder=hp_grounder))
    client = chromadb.PersistentClient(path=str(built_db))
    collection = client.get_collection(COLLECTION_NAME)

    assert collection.count() == expected
    assert collection.metadata["hnsw:space"] == "cosine"


def test_build_database_embeds_unique_strings_once(snomed_root, hp_grounder, encoder, tmp_path):
    # A spy that records what gets embedded, delegating to the real encoder.
    class RecordingEncoder:
        def __init__(self, inner):
            self.inner = inner
            self.calls: list[list[str]] = []

        def encode(self, texts, *args, **kwargs):
            self.calls.append(list(texts))
            return self.inner.encode(texts, *args, **kwargs)

    spy = RecordingEncoder(encoder)
    build_chromadb.build_database(
        snomed_root=snomed_root,
        chroma_path=tmp_path / "chroma",
        encoder=spy,  # type: ignore[arg-type]  # duck-typed spy, only .encode is used
        hp_grounder=hp_grounder,
    )

    # Exactly one encode() call, over the deduplicated lowercased strings.
    assert len(spy.calls) == 1
    lowered = [t.lower() for t in spy.calls[0]]
    assert len(lowered) == len(set(lowered))


def test_build_database_refuses_existing_without_rebuild(built_db, snomed_root, hp_grounder, encoder):
    with pytest.raises(SystemExit, match="already exists"):
        build_chromadb.build_database(
            snomed_root=snomed_root, chroma_path=built_db,
            encoder=encoder, hp_grounder=hp_grounder,
        )


def test_build_database_rebuild_overwrites(built_db, snomed_root, hp_grounder, encoder):
    import chromadb

    build_chromadb.build_database(
        snomed_root=snomed_root, chroma_path=built_db, rebuild=True,
        encoder=encoder, hp_grounder=hp_grounder,
    )

    client = chromadb.PersistentClient(path=str(built_db))
    expected = len(collect_terms(snomed_root, hp_grounder=hp_grounder))
    assert client.get_collection(COLLECTION_NAME).count() == expected


def test_build_database_erase_removes_directory(built_db, snomed_root, hp_grounder, encoder):
    import chromadb
    from chromadb.api.shared_system_client import SharedSystemClient

    # Drop a stray marker file to prove the whole directory is erased.
    marker = built_db / "leftover.txt"
    marker.write_text("stale", encoding="utf-8")

    # ChromaDB caches a System per path within a process; a real CLI --erase run
    # is a fresh process, so clear the cache to reproduce that here.
    SharedSystemClient.clear_system_cache()

    build_chromadb.build_database(
        snomed_root=snomed_root, chroma_path=built_db, erase=True,
        encoder=encoder, hp_grounder=hp_grounder,
    )

    assert not marker.exists()
    client = chromadb.PersistentClient(path=str(built_db))
    expected = len(collect_terms(snomed_root, hp_grounder=hp_grounder))
    assert client.get_collection(COLLECTION_NAME).count() == expected


def test_build_database_no_terms_raises(empty_snomed_root, encoder, tmp_path):
    empty_hp = Grounder(terms=[])
    with pytest.raises(SystemExit, match="No terms"):
        build_chromadb.build_database(
            snomed_root=empty_snomed_root, chroma_path=tmp_path / "chroma",
            encoder=encoder, hp_grounder=empty_hp,
        )


# =========================================================================== #
# Query: IndexQueryUtil / load_semantic_grounder
# =========================================================================== #
def test_ground_returns_candidates_best_first(built_db, encoder):
    util = IndexQueryUtil(chroma_path=built_db, embedder=encoder)

    results = util.ground(["asthma", "nephrostomy"], top_k=3)

    assert len(results) == 2  # one candidate list per query
    for cands in results:
        assert len(cands) <= 3
        assert all(isinstance(c, Candidate) for c in cands)
        # Scores are cosine similarities in [0, 1], sorted best-first.
        scores = [c.score for c in cands]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)


def test_ground_retrieves_expected_concept(built_db, encoder):
    util = IndexQueryUtil(chroma_path=built_db, embedder=encoder)

    top = util.ground(["asthma"], top_k=1)[0][0]

    # The nearest surface form is an asthma concept; identifier is "<db>:<id>".
    assert top.concept_id == ASTHMA
    assert top.db == "SNOMEDCT_disorder"
    assert top.identifier == f"SNOMEDCT_disorder:{ASTHMA}"
    assert top.text in {ASTHMA_FSN, ASTHMA_SYN}
    assert top.name == ASTHMA_FSN


def test_ground_retrieves_hp_concept(built_db, encoder):
    util = IndexQueryUtil(chroma_path=built_db, embedder=encoder)

    top = util.ground(["difficulty breathing"], top_k=1)[0][0]

    assert top.db == "HP"
    assert top.concept_id == "HP:0002098"
    assert top.identifier == "HP:HP:0002098"


def test_ground_empty_queries_returns_empty(built_db, encoder):
    util = IndexQueryUtil(chroma_path=built_db, embedder=encoder)
    assert util.ground([]) == []


def test_ground_respects_top_k(built_db, encoder):
    util = IndexQueryUtil(chroma_path=built_db, embedder=encoder)
    (cands,) = util.ground(["fever"], top_k=2)
    assert len(cands) == 2


def test_index_query_util_missing_collection_raises(tmp_path, encoder):
    with pytest.raises(SystemExit, match="Could not open collection"):
        IndexQueryUtil(chroma_path=tmp_path / "empty", embedder=encoder)


def test_load_semantic_grounder_builds_query_util(built_db, encoder):
    grounder = load_semantic_grounder(chroma_path=built_db, embedder=encoder)

    assert isinstance(grounder, IndexQueryUtil)
    top = grounder.ground(["fever"], top_k=1)[0][0]
    assert top.concept_id == FEVER
    assert top.db == "SNOMEDCT_finding"
