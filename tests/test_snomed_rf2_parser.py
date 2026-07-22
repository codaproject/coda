"""Tests for the SNOMED CT RF2 parser (:mod:`coda.snomed.rf2`).

A tiny synthetic RF2 release tree is built under ``tmp_path`` for each test, so
the parser is exercised end-to-end (concept/description/language-refset loading
-> surface-form iteration) without the multi-gigabyte real release (which is
git-ignored). Safe to run on CI.

The GILDA grounder no longer parses RF2 directly — it sources SNOMED CT terms
from the CODA KG (see ``test_snomed_kg_grounder.py``). The RF2 parsing here is
the shared source of truth feeding the KG exporter.

The synthetic fixture packs one example of every branch the parser cares about:
active vs. inactive concepts and descriptions, FSN vs. synonym vs. other
description types, relevant vs. irrelevant semantic-tag categories, and
synonyms that are Preferred, Acceptable, absent from the refset, in the wrong
refset, or carry an unrecognized acceptability.
"""
from pathlib import Path

import pytest

from coda.snomed.rf2 import (
    ACCEPTABLE_ID,
    FSN_TYPE_ID,
    PREFERRED_ID,
    SYNONYM_TYPE_ID,
    US_ENGLISH_REFSET_ID,
    find_snapshot_file,
    iter_surface_forms,
    load_active_concepts,
    load_descriptions,
    load_language_refset,
)

# --- RF2 metadata constants used only to make the fixture rows realistic ---
MODULE_ID = "900000000000207008"
CASE_ID = "900000000000448009"
DEF_STATUS_ID = "900000000000074008"
DEFINITION_TYPE_ID = "900000000000550004"  # a description type that is not FSN/synonym
GB_ENGLISH_REFSET_ID = "900000000000508004"  # a refset that is not US English
BOGUS_ACCEPTABILITY_ID = "111111111111111111"
EFFECTIVE = "20260601"

CONCEPT_HEADER = ["id", "effectiveTime", "active", "moduleId", "definitionStatusId"]
DESCRIPTION_HEADER = [
    "id", "effectiveTime", "active", "moduleId", "conceptId",
    "languageCode", "typeId", "term", "caseSignificanceId",
]
LANGUAGE_HEADER = [
    "id", "effectiveTime", "active", "moduleId",
    "refsetId", "referencedComponentId", "acceptabilityId",
]


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(header)]
    lines += ["\t".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _concept(cid: str, active: str) -> list[str]:
    return [cid, EFFECTIVE, active, MODULE_ID, DEF_STATUS_ID]


def _description(
    desc_id: str, concept_id: str, type_id: str, term: str, active: str = "1"
) -> list[str]:
    return [
        desc_id, EFFECTIVE, active, MODULE_ID, concept_id,
        "en", type_id, term, CASE_ID,
    ]


def _language(
    member_id: str, desc_id: str, acceptability_id: str, refset_id: str = US_ENGLISH_REFSET_ID
) -> list[str]:
    return [member_id, EFFECTIVE, "1", MODULE_ID, refset_id, desc_id, acceptability_id]


# Concept ids used throughout the fixture.
ASTHMA = "100000001"       # active disorder, exercises synonym acceptability branches
NEPHROSTOMY = "100000002"  # active procedure
FEVER = "100000003"        # active finding, FSN only (no synonyms)
CAFFEINE = "100000004"     # active but "substance" -> irrelevant category, fully excluded
HISTORICAL = "100000005"   # inactive concept, everything on it excluded


@pytest.fixture
def snomed_root(tmp_path: Path) -> Path:
    """Build a synthetic RF2 release tree and return its root directory."""
    root = tmp_path / "release"
    terminology = root / "Snapshot" / "Terminology"
    language = root / "Snapshot" / "Refset" / "Language"

    _write_tsv(
        terminology / f"sct2_Concept_Snapshot_INT_{EFFECTIVE}.txt",
        CONCEPT_HEADER,
        [
            _concept(ASTHMA, "1"),
            _concept(NEPHROSTOMY, "1"),
            _concept(FEVER, "1"),
            _concept(CAFFEINE, "1"),
            _concept(HISTORICAL, "0"),  # inactive
        ],
    )

    _write_tsv(
        terminology / f"sct2_Description_Snapshot-en_INT_{EFFECTIVE}.txt",
        DESCRIPTION_HEADER,
        [
            # --- Asthma: FSN + a spread of synonym cases ---
            _description("200000001", ASTHMA, FSN_TYPE_ID, "Asthma (disorder)"),
            _description("200000002", ASTHMA, SYNONYM_TYPE_ID, "Asthma"),                # preferred
            _description("200000003", ASTHMA, SYNONYM_TYPE_ID, "Bronchial asthma"),      # acceptable
            _description("200000004", ASTHMA, SYNONYM_TYPE_ID, "Asthma NOS"),            # no refset row
            _description("200000005", ASTHMA, SYNONYM_TYPE_ID, "Reactive airway disease"),  # wrong refset
            _description("200000006", ASTHMA, SYNONYM_TYPE_ID, "Wheezy bronchitis"),     # bogus acceptability
            _description("200000007", ASTHMA, SYNONYM_TYPE_ID, "Inactive synonym", active="0"),
            _description("200000008", ASTHMA, DEFINITION_TYPE_ID, "A definition text"),  # non-FSN/synonym type
            # --- Nephrostomy: FSN + one preferred synonym ---
            _description("200000010", NEPHROSTOMY, FSN_TYPE_ID, "Nephrostomy (procedure)"),
            _description("200000011", NEPHROSTOMY, SYNONYM_TYPE_ID, "Nephrostomy"),
            # --- Fever: FSN only ---
            _description("200000020", FEVER, FSN_TYPE_ID, "Fever (finding)"),
            # --- Caffeine: irrelevant category, so nothing should surface ---
            _description("200000030", CAFFEINE, FSN_TYPE_ID, "Caffeine (substance)"),
            _description("200000031", CAFFEINE, SYNONYM_TYPE_ID, "Caffeine"),
            # --- Historical: on an inactive concept, excluded ---
            _description("200000040", HISTORICAL, FSN_TYPE_ID, "Historical disorder (disorder)"),
        ],
    )

    _write_tsv(
        language / f"der2_cRefset_LanguageSnapshot-en_INT_{EFFECTIVE}.txt",
        LANGUAGE_HEADER,
        [
            _language("300000001", "200000002", PREFERRED_ID),
            _language("300000002", "200000003", ACCEPTABLE_ID),
            _language("300000003", "200000005", ACCEPTABLE_ID, refset_id=GB_ENGLISH_REFSET_ID),
            _language("300000004", "200000006", BOGUS_ACCEPTABILITY_ID),
            _language("300000005", "200000011", PREFERRED_ID),
            _language("300000006", "200000031", PREFERRED_ID),  # on the excluded caffeine concept
            _language("300000007", "200000001", PREFERRED_ID),  # FSN also has a refset row
        ],
    )

    return root


# --------------------------------------------------------------------------- #
# _find_snapshot_file
# --------------------------------------------------------------------------- #
def test_find_snapshot_file_picks_latest(tmp_path: Path):
    (tmp_path / "sct2_Concept_Snapshot_INT_20250101.txt").write_text("", encoding="utf-8")
    (tmp_path / "sct2_Concept_Snapshot_INT_20260601.txt").write_text("", encoding="utf-8")

    found = find_snapshot_file(tmp_path, "sct2_Concept_Snapshot_*.txt")

    # Sorted lexicographically, the newer date wins.
    assert found.name == "sct2_Concept_Snapshot_INT_20260601.txt"


def test_find_snapshot_file_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No file matching"):
        find_snapshot_file(tmp_path, "sct2_Concept_Snapshot_*.txt")


# --------------------------------------------------------------------------- #
# _load_active_concepts
# --------------------------------------------------------------------------- #
def test_load_active_concepts_excludes_inactive(snomed_root: Path):
    terminology = snomed_root / "Snapshot" / "Terminology"
    active = load_active_concepts(terminology)

    assert active == {ASTHMA, NEPHROSTOMY, FEVER, CAFFEINE}
    assert HISTORICAL not in active


# --------------------------------------------------------------------------- #
# _load_descriptions
# --------------------------------------------------------------------------- #
def test_load_descriptions_filters_and_tags_category(snomed_root: Path):
    terminology = snomed_root / "Snapshot" / "Terminology"
    active = load_active_concepts(terminology)
    descriptions = load_descriptions(terminology, active)

    # Inactive description, the definition-type description, and the description
    # on the inactive concept are all dropped.
    assert "200000007" not in descriptions  # inactive synonym
    assert "200000008" not in descriptions  # definition type
    assert "200000040" not in descriptions  # on inactive concept

    # Synonyms are kept even for the irrelevant category here; category filtering
    # happens later in iter_surface_forms.
    assert "200000030" in descriptions  # caffeine FSN survives description loading

    # FSN semantic tags are parsed out into the category field.
    assert descriptions["200000001"]["category"] == "disorder"
    assert descriptions["200000010"]["category"] == "procedure"
    assert descriptions["200000020"]["category"] == "finding"
    # Synonyms carry no category tag.
    assert descriptions["200000002"]["category"] == ""


# --------------------------------------------------------------------------- #
# _load_language_refset
# --------------------------------------------------------------------------- #
def test_load_language_refset_only_us_english(snomed_root: Path):
    acceptability = load_language_refset(snomed_root / "Snapshot")

    assert acceptability["200000002"] == PREFERRED_ID
    assert acceptability["200000003"] == ACCEPTABLE_ID
    assert acceptability["200000006"] == BOGUS_ACCEPTABILITY_ID
    # The GB-English refset row is ignored entirely.
    assert "200000005" not in acceptability


# --------------------------------------------------------------------------- #
# iter_surface_forms (FSN + synonym surface forms, with filtering)
# --------------------------------------------------------------------------- #
def test_iter_surface_forms_emits_expected(snomed_root: Path):
    got = {(sf.text, sf.status) for sf in iter_surface_forms(snomed_root)}
    assert got == {
        ("Asthma (disorder)", "name"),
        ("Asthma", "synonym"),
        ("Bronchial asthma", "synonym"),
        ("Nephrostomy (procedure)", "name"),
        ("Nephrostomy", "synonym"),
        ("Fever (finding)", "name"),
    }


def test_iter_surface_forms_excludes_irrelevant_and_inactive(snomed_root: Path):
    texts = {sf.text for sf in iter_surface_forms(snomed_root)}

    # Irrelevant "substance" category is dropped despite being active and in refset.
    assert "Caffeine (substance)" not in texts
    assert "Caffeine" not in texts
    # Everything on the inactive concept is gone.
    assert "Historical disorder (disorder)" not in texts
    # Synonyms not in the US-English refset (missing, wrong refset, bad acceptability) are dropped.
    assert "Asthma NOS" not in texts
    assert "Reactive airway disease" not in texts
    assert "Wheezy bronchitis" not in texts


def test_iter_surface_forms_attributes(snomed_root: Path):
    by_text = {sf.text: sf for sf in iter_surface_forms(snomed_root)}

    fsn = by_text["Asthma (disorder)"]
    assert fsn.status == "name"
    assert fsn.category == "disorder"
    assert fsn.concept_id == ASTHMA
    assert fsn.entry_name == "Asthma (disorder)"

    # A synonym inherits its concept id, category, and FSN-derived entry_name.
    syn = by_text["Bronchial asthma"]
    assert syn.status == "synonym"
    assert syn.concept_id == ASTHMA
    assert syn.category == "disorder"
    assert syn.entry_name == "Asthma (disorder)"

    # Different concepts carry their own semantic-tag category.
    assert by_text["Nephrostomy (procedure)"].category == "procedure"
    assert by_text["Fever (finding)"].category == "finding"
