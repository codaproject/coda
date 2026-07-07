"""Build a GILDA grounder from a SNOMED CT RF2 release.

The framework-neutral RF2 parsing lives in :mod:`coda.snomed.rf2`; this module
adds the gilda-specific layer on top (constructing :class:`gilda.Term` objects
and a :class:`~gilda.grounder.Grounder`). The parsing primitives and RF2
constants are re-exported here under their historical names so existing
imports keep working.
"""
from pathlib import Path

from gilda import Term, make_grounder
from gilda.grounder import Grounder
from gilda.process import normalize

# Several names are re-exported (not used here) to preserve this module's
# historical public API, which tests and callers still import.
from coda.snomed.rf2 import (  # noqa: F401
    ACCEPTABLE_ID,
    FSN_TYPE_ID,
    PREFERRED_ID,
    RELEVANT_CATEGORIES,
    SYNONYM_TYPE_ID,
    US_ENGLISH_REFSET_ID,
    find_snapshot_file as _find_snapshot_file,
    iter_surface_forms,
    load_active_concepts as _load_active_concepts,
    load_descriptions as _load_descriptions,
    load_language_refset as _load_language_refset,
)

SOURCE = "SNOMED CT RF2 International Release"


def _parse_rf2_terms(snomed_root: Path) -> list[Term]:
    """
    Parse a SNOMED CT RF2 release tree rooted at `snomed_root` and return
    GILDA Term objects for all active concepts.

    Each concept produces:
      - One Term (status="name") for its FSN
      - One or more Terms (status="synonym") for every active US-English synonym
        (both Preferred and Acceptable acceptability)
    """
    terms: list[Term] = []
    for sf in iter_surface_forms(snomed_root, RELEVANT_CATEGORIES):
        norm = normalize(sf.text)
        if not norm:
            continue
        db = f"SNOMEDCT_{sf.category.replace(' ', '_')}"
        terms.append(Term(
            norm_text=norm,
            text=sf.text,
            db=db,
            id=sf.concept_id,
            entry_name=sf.entry_name,
            status=sf.status,
            source=SOURCE,
        ))

    return terms

def make_gilda_grounder(snomed_root: Path) -> Grounder:
    """Build a GILDA Grounder from a SNOMED CT RF2 release directory."""
    terms = _parse_rf2_terms(snomed_root)
    return make_grounder(terms)
