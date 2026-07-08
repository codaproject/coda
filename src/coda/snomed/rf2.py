"""Framework-neutral parser for the SNOMED CT RF2 release format.

These helpers walk an RF2 ``Snapshot`` tree and return plain Python data
structures (sets, dicts, tuples). They intentionally have no dependency on
gilda, pandas, or Neo4j. They feed the knowledge-graph exporter
(:mod:`coda.kg.sources.snomedct`), which imports SNOMED CT into the CODA KG;
the GILDA-based event grounder then sources its terms from the KG at runtime
(:mod:`coda.grounding.temporal_ordering.event_grounding.snomed_kg_utils`).

A SNOMED CT RF2 release is distributed under license, so the data itself is
never version controlled; callers point these functions at a locally provided
release directory.
"""
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# The description ``term`` column can be very long; without this the RF2 files
# fail to parse on some concepts.
csv.field_size_limit(sys.maxsize)


# RF2 description type concept IDs
FSN_TYPE_ID = "900000000000003001"
SYNONYM_TYPE_ID = "900000000000013009"

# US English language reference set
US_ENGLISH_REFSET_ID = "900000000000509007"

# Acceptability concept IDs
PREFERRED_ID = "900000000000548007"
ACCEPTABLE_ID = "900000000000549004"

# RF2 relationship type concept ID for the "Is a" hierarchy.
IS_A_TYPE_ID = "116680003"

# FSN semantic tags kept by both the grounder and the KG exporter. Keeping the
# scope identical means the two subsystems stay in lockstep.
RELEVANT_CATEGORIES = {"disorder", "procedure", "finding"}

# Captures the trailing "(<semantic tag>)" of a Fully Specified Name.
_FSN_TAG_RE = re.compile(r"^(.*?)\s*\(([^)]+)\)\s*$")


def find_snapshot_file(snapshot_dir: Path, pattern: str) -> Path:
    """Return the newest RF2 file matching ``pattern`` under ``snapshot_dir``."""
    matches = list(snapshot_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' found under {snapshot_dir}")
    return sorted(matches)[-1]


def load_active_concepts(terminology_dir: Path) -> set[str]:
    """Return the set of active concept IDs."""
    path = find_snapshot_file(terminology_dir, "sct2_Concept_Snapshot_*.txt")
    active = set()
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["active"] == "1":
                active.add(row["id"])
    return active


def load_descriptions(terminology_dir: Path, active_concepts: set[str]) -> dict[str, dict]:
    """Return {description_id: {conceptId, typeId, term, category}} for active
    FSN/synonym descriptions attached to active concepts."""
    path = find_snapshot_file(terminology_dir, "sct2_Description_Snapshot-en_*.txt")
    descriptions: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["active"] != "1":
                continue
            if row["conceptId"] not in active_concepts:
                continue
            if row["typeId"] not in (FSN_TYPE_ID, SYNONYM_TYPE_ID):
                continue
            category = ""
            if row["typeId"] == FSN_TYPE_ID:
                m = _FSN_TAG_RE.match(row["term"])
                category = m.group(2).strip() if m else ""
            descriptions[row["id"]] = {
                "conceptId": row["conceptId"],
                "typeId": row["typeId"],
                "term": row["term"],
                "category": category,
            }
    return descriptions


def load_language_refset(snapshot_dir: Path) -> dict[str, str]:
    """Return {description_id: acceptabilityId} for the US English refset."""
    lang_dir = snapshot_dir / "Refset" / "Language"
    path = find_snapshot_file(lang_dir, "der2_cRefset_LanguageSnapshot-en_*.txt")
    acceptability: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["active"] != "1":
                continue
            if row["refsetId"] != US_ENGLISH_REFSET_ID:
                continue
            acceptability[row["referencedComponentId"]] = row["acceptabilityId"]
    return acceptability


def load_isa_relationships(
    terminology_dir: Path, valid_concepts: set[str]
) -> list[tuple[str, str]]:
    """Return active ``Is a`` edges as (sourceId, destinationId) tuples.

    Only relationships whose both endpoints are in ``valid_concepts`` are kept,
    so the resulting hierarchy stays within the exported concept set.
    """
    path = find_snapshot_file(terminology_dir, "sct2_Relationship_Snapshot_*.txt")
    edges: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["active"] != "1":
                continue
            if row["typeId"] != IS_A_TYPE_ID:
                continue
            source_id = row["sourceId"]
            destination_id = row["destinationId"]
            if source_id not in valid_concepts or destination_id not in valid_concepts:
                continue
            edges.append((source_id, destination_id))
    return edges


def index_concepts(descriptions: dict[str, dict]) -> tuple[dict[str, str], dict[str, str]]:
    """Build per-concept FSN and semantic-tag lookups from parsed descriptions.

    Returns ``(concept_fsn, concept_category)`` mapping each concept ID to its
    Fully Specified Name and to the semantic tag parsed from that FSN.
    """
    concept_fsn: dict[str, str] = {}
    concept_category: dict[str, str] = {}
    for desc in descriptions.values():
        if desc["typeId"] == FSN_TYPE_ID:
            cid = desc["conceptId"]
            concept_fsn[cid] = desc["term"]
            concept_category[cid] = desc["category"]
    return concept_fsn, concept_category


@dataclass
class SurfaceForm:
    """One grounding surface form (FSN or synonym) of a SNOMED concept."""

    description_id: str  # RF2 description SCTID (its own, distinct from concept_id)
    concept_id: str      # the concept this surface form describes
    text: str            # the surface-form string itself
    status: str          # "name" (the FSN) | "synonym"
    category: str        # the concept's semantic tag (disorder/procedure/finding)
    entry_name: str      # the concept's FSN (canonical name)


def iter_surface_forms(
    snomed_root: Path, categories: set[str] = RELEVANT_CATEGORIES
) -> Iterator[SurfaceForm]:
    """Yield the grounding surface forms of every active concept in ``categories``.

    For each such concept this yields one :class:`SurfaceForm` with ``status``
    ``"name"`` for the FSN, plus one ``"synonym"`` for every active US-English
    synonym with Preferred or Acceptable acceptability. This is the single
    source of truth for "which surface forms count", shared by the GILDA
    grounder and the knowledge-graph exporter.
    """
    snapshot_dir = snomed_root / "Snapshot"
    terminology_dir = snapshot_dir / "Terminology"

    active_concepts = load_active_concepts(terminology_dir)
    descriptions = load_descriptions(terminology_dir, active_concepts)
    acceptability = load_language_refset(snapshot_dir)
    concept_fsn, concept_category = index_concepts(descriptions)

    for desc_id, desc in descriptions.items():
        cid = desc["conceptId"]
        category = concept_category.get(cid)
        if category not in categories:
            continue

        if desc["typeId"] == FSN_TYPE_ID:
            status = "name"
        else:
            # Only surface synonyms in the US-English refset (Preferred/Acceptable).
            if acceptability.get(desc_id) not in (PREFERRED_ID, ACCEPTABLE_ID):
                continue
            status = "synonym"

        yield SurfaceForm(
            description_id=desc_id,
            concept_id=cid,
            text=desc["term"],
            status=status,
            category=category,
            entry_name=concept_fsn.get(cid, desc["term"]),
        )
