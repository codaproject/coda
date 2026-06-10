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
    KG_BASE,
    REPORTS_BASE,
    KGSourceExporter,
)
from coda.kg.embed_nodes import embed_nodes
from coda.kg.processor_util import check_duplicated_nodes, \
    check_missing_node_ids_in_edges
from coda.kg.reports import generate_reports

# Sources that don't pre-compute embeddings during export and need a
# post-export embedding pass.
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
    for exporter in EXPORTERS:
        if exporter.name in EMBED_SOURCES:
            embed_nodes(exporter.nodes_file)
    check_duplicated_nodes(exporters=EXPORTERS, strict=False)
    check_missing_node_ids_in_edges(exporters=EXPORTERS, strict=False)
    generate_reports(EXPORTERS, build_seconds=time.time() - start)


if __name__ == "__main__":
    dump_kg()
