
from pathlib import Path

from gilda import Term, make_grounder
from gilda.grounder import Grounder
from gilda.process import normalize

import csv, re, sys

# This is necessary to be able to read the SNOMED CT files correctly
csv.field_size_limit(sys.maxsize)


# RF2 type concept IDs
FSN_TYPE_ID = "900000000000003001"
SYNONYM_TYPE_ID = "900000000000013009"

# US English language reference set
US_ENGLISH_REFSET_ID = "900000000000509007"

# Acceptability concept IDs
PREFERRED_ID = "900000000000548007"
ACCEPTABLE_ID = "900000000000549004"

_FSN_TAG_RE = re.compile(r"^(.*?)\s*\(([^)]+)\)\s*$")

SOURCE = "SNOMED CT RF2 International Release"


def _find_snapshot_file(snapshot_dir: Path, pattern: str) -> Path:
    matches = list(snapshot_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' found under {snapshot_dir}")
    return sorted(matches)[-1]


def _load_active_concepts(terminology_dir: Path) -> set[str]:
    path = _find_snapshot_file(terminology_dir, "sct2_Concept_Snapshot_*.txt")
    active = set()
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["active"] == "1":
                active.add(row["id"])
    return active


def _load_descriptions(terminology_dir: Path, active_concepts: set[str]) -> dict[str, dict]:
    """Returns {description_id: {conceptId, typeId, term, category}} for active descriptions on active concepts."""
    path = _find_snapshot_file(terminology_dir, "sct2_Description_Snapshot-en_*.txt")
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


def _load_language_refset(snapshot_dir: Path) -> dict[str, str]:
    """Returns {description_id: acceptabilityId} for the US English refset."""
    lang_dir = snapshot_dir / "Refset" / "Language"
    path = _find_snapshot_file(lang_dir, "der2_cRefset_LanguageSnapshot-en_*.txt")
    acceptability: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["active"] != "1":
                continue
            if row["refsetId"] != US_ENGLISH_REFSET_ID:
                continue
            acceptability[row["referencedComponentId"]] = row["acceptabilityId"]
    return acceptability

def _parse_rf2_terms(snomed_root: Path) -> list[Term]:
    """
    Parse a SNOMED CT RF2 release tree rooted at `snomed_root` and return
    GILDA Term objects for all active concepts.

    Each concept produces:
      - One Term (status="name") for its FSN
      - One or more Terms (status="synonym") for every active US-English synonym
        (both Preferred and Acceptable acceptability)
    """
    snapshot_dir = snomed_root / "Snapshot"
    terminology_dir = snapshot_dir / "Terminology"

    active_concepts = _load_active_concepts(terminology_dir)
    descriptions = _load_descriptions(terminology_dir, active_concepts)
    acceptability = _load_language_refset(snapshot_dir)

    # Build a per-concept FSN lookup for entry_name
    concept_fsn: dict[str, str] = {}
    concept_category: dict[str, str] = {}
    for desc in descriptions.values():
        if desc["typeId"] == FSN_TYPE_ID:
            cid = desc["conceptId"]
            concept_fsn[cid] = desc["term"]
            concept_category[cid] = desc["category"]

    RELEVANT_CATEGORIES = {"disorder", "procedure", "finding"}

    terms: list[Term] = []
    for desc_id, desc in descriptions.items():
        cid = desc["conceptId"]
        if concept_category.get(cid) not in RELEVANT_CATEGORIES:
            continue
        text = desc["term"]
        norm = normalize(text)
        if not norm:
            continue

        db = f"SNOMEDCT_{concept_category.get(cid, 'unknown').replace(' ', '_')}"
        entry_name = concept_fsn.get(cid, text)

        if desc["typeId"] == FSN_TYPE_ID:
            terms.append(Term(
                norm_text=norm,
                text=text,
                db=db,
                id=cid,
                entry_name=entry_name,
                status="name",
                source=SOURCE,
            ))
        else:
            # Only emit synonym Terms for descriptions in the US English refset
            acc = acceptability.get(desc_id)
            if acc not in (PREFERRED_ID, ACCEPTABLE_ID):
                continue
            terms.append(Term(
                norm_text=norm,
                text=text,
                db=db,
                id=cid,
                entry_name=entry_name,
                status="synonym",
                source=SOURCE,
            ))

    return terms

def make_rf2_grounder(snomed_root: Path) -> Grounder:
    """Build a GILDA Grounder from a SNOMED CT RF2 release directory."""
    terms = _parse_rf2_terms(snomed_root)
    return make_grounder(terms)
