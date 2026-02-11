from tqdm import tqdm

from coda.kg.sources import (
    icd10,
    icd11,
    phmrc,
    who_va,
    acme,
    probbase,
    hpo,
    KG_BASE,
    KGSourceExporter,
)
from coda.kg.processor_util import (
    check_duplicated_nodes,
    check_missing_node_ids_in_edges,
)


EXPORTERS: list[KGSourceExporter] = [
    icd10.ICD10Exporter(),
    icd11.ICD11Exporter(),
    phmrc.PhmrcExporter(),
    who_va.WhoVaExporter(),
    acme.ACMEExporter(),
    probbase.ProbBaseExporter(),
    hpo.HpoExporter(),
]


def dump_kg():
    """Dump the knowledge graph to file."""
    # Make folder if needed
    KG_BASE.mkdir(exist_ok=True)

    for exporter in tqdm(
        EXPORTERS,
        desc="Exporting KG sources",
        unit="source",
    ):
        exporter.export()
    check_duplicated_nodes(exporters=EXPORTERS, strict=False)
    check_missing_node_ids_in_edges(exporters=EXPORTERS, strict=True)


if __name__ == "__main__":
    dump_kg()
