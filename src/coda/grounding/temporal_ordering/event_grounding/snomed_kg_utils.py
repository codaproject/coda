"""Build a GILDA grounder from the SNOMED CT terms stored in the CODA KG.

Historically this grounder parsed a SNOMED CT RF2 release from disk. The RF2
release is now imported into the CODA KG as ``snomedct`` nodes (one per surface
form, FSN + synonyms) by :mod:`coda.kg.sources.snomedct`, so the grounder can
source the exact same terms from the KG at startup instead of re-parsing the
licensed release. This decouples runtime grounding from the RF2 data path: the
only thing needed at runtime is the KG URL (see :func:`coda.runtime_config.get_kg_url`),
exactly like the sibling SapBERT grounder and the RAG retriever.

The framework-neutral RF2 parsing still lives in :mod:`coda.snomed.rf2`; it is
used by the KG exporter, not here.
"""
import logging

from gilda import Term, make_grounder
from gilda.grounder import Grounder
from gilda.process import normalize
from neo4j import GraphDatabase

from coda.runtime_config import get_kg_url

logger = logging.getLogger(__name__)

SOURCE = "SNOMED CT (via CODA KG)"

# Node label under which SNOMED CT surface forms are imported into the KG. Every
# node (FSN and synonym alike) carries the fields needed to build a gilda Term.
# Keep in sync with coda.kg.sources.snomedct.
_SNOMED_LABEL = "snomedct"


def _fetch_snomed_terms(driver) -> list[Term]:
    """Query all ``snomedct`` surface-form nodes and build GILDA Terms.

    Each KG node maps to one Term, matching the historical RF2-derived shape:
      - ``db``  = ``SNOMEDCT_<category>`` (semantic tag, spaces -> underscores)
      - ``id``  = the concept id (``code``)
      - ``entry_name`` / ``status`` / ``text`` = the node's fields
    FSN nodes carry ``status="name"`` and synonym nodes ``status="synonym"``,
    just as :func:`coda.snomed.rf2.iter_surface_forms` produced them.
    """
    terms: list[Term] = []
    with driver.session() as session:
        records = session.run(
            f"""
            MATCH (n:`{_SNOMED_LABEL}`)
            RETURN n.text AS text, n.category AS category, n.code AS code,
                   n.name AS name, n.status AS status
            """
        )
        for record in records:
            text = record["text"]
            norm = normalize(text) if text else ""
            if not norm:
                continue
            category = record["category"] or ""
            db = f"SNOMEDCT_{category.replace(' ', '_')}"
            terms.append(Term(
                norm_text=norm,
                text=text,
                db=db,
                id=str(record["code"]),
                entry_name=record["name"],
                status=record["status"],
                source=SOURCE,
            ))

    return terms


def snomedct_terms_available(kg_url: str | None = None) -> bool:
    """Return True if the KG contains any ``snomedct`` nodes.

    Used as the runtime enablement gate for event grounding: SNOMED CT is
    licensed and only present in the KG when a release was available at build
    time, so an empty result means grounding should stay disabled. Since event
    grounding is optional, an unreachable KG is treated the same as an empty one
    (returns False) rather than raising, so importing modules that gate on this
    never crash when the KG is down.
    """
    try:
        driver = GraphDatabase.driver(kg_url or get_kg_url(), auth=None)
    except Exception:  # noqa: BLE001 -- any driver/config error means "unavailable"
        logger.warning("Could not open a KG connection; event grounding disabled", exc_info=True)
        return False
    try:
        with driver.session() as session:
            record = session.run(
                f"MATCH (n:`{_SNOMED_LABEL}`) RETURN count(n) > 0 AS present"
            ).single()
            return bool(record["present"]) if record is not None else False
    except Exception:  # noqa: BLE001 -- KG unreachable/query error -> grounding off
        logger.warning("KG unreachable while probing SNOMED CT; event grounding disabled", exc_info=True)
        return False
    finally:
        driver.close()


def make_gilda_grounder(kg_url: str | None = None) -> Grounder:
    """Build a GILDA Grounder from the SNOMED CT terms in the CODA KG."""
    driver = GraphDatabase.driver(kg_url or get_kg_url(), auth=None)
    try:
        terms = _fetch_snomed_terms(driver)
    finally:
        driver.close()
    logger.info("Loaded %d SNOMED CT terms from the KG", len(terms))
    return make_grounder(terms)
