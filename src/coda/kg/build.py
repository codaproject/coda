import logging
import time

from tqdm import tqdm

from coda.kg.sources import (
    icd10,
    icd11,
    phmrc,
    who_va,
    acme,
    probbase,
    hpo,
    mesh,
    mondo,
    wdi,
    who_mortality,
    snomedct,
    KG_BASE,
    REPORTS_BASE,
    KGSourceExporter,
)
from coda.kg.processor_util import check_duplicated_nodes, \
    check_missing_node_ids_in_edges
from coda.kg.reports import generate_reports

logger = logging.getLogger(__name__)

# Sources that don't pre-compute embeddings during export and need a
# post-export embedding pass. (snomedct embeds its own SapBERT vectors inside
# export(), so it is intentionally not listed here.)
EMBED_SOURCES = ["icd11"]


EXPORTERS: list[KGSourceExporter] = [
    icd10.ICD10Exporter(),
    icd11.ICD11Exporter(),
    phmrc.PhmrcExporter(),
    who_va.WhoVaExporter(),
    acme.ACMEExporter(),
    probbase.ProbBaseExporter(),
    hpo.HpoExporter(),
    mesh.MeshExporter(),
    mondo.MondoExporter(),
    wdi.WDIExporter(),
    who_mortality.WhoMortalityExporter(),
]

# SNOMED CT is licensed and optional: only export it when a release directory
# is configured (SNOMED_DATA_PATH or the temporal-ordering config).
_snomed_root = snomedct.resolve_snomed_root()
if _snomed_root is not None:
    logger.info("SNOMED CT data found at %s; including it in the build", _snomed_root)
    EXPORTERS.append(snomedct.SnomedCtExporter(_snomed_root))
else:
    logger.info("No SNOMED CT data configured; skipping the SNOMED source")


def dump_kg():
    """Dump the knowledge graph to file."""
    # Make folders if needed
    KG_BASE.mkdir(exist_ok=True)
    REPORTS_BASE.mkdir(parents=True, exist_ok=True)

    start = time.time()
    for exporter in tqdm(
        EXPORTERS,
        desc="Exporting KG sources",
        unit="source",
    ):
        exporter.export()

    # Precompute embeddings for sources that need a post-export pass. Imported
    # lazily so the sentence-transformers stack is only loaded when embedding
    # is actually required.
    embed_exporters = [e for e in EXPORTERS if e.name in EMBED_SOURCES]
    if embed_exporters:
        from coda.kg.embed_nodes import embed_nodes
        for exporter in tqdm(
            embed_exporters, desc="Embedding node names", unit="source"
        ):
            embed_nodes(exporter.nodes_file)

    check_duplicated_nodes(exporters=EXPORTERS, strict=False)
    check_missing_node_ids_in_edges(exporters=EXPORTERS, strict=False)
    generate_reports(EXPORTERS, build_seconds=time.time() - start)


if __name__ == "__main__":
    dump_kg()
