"""SNOMED CT knowledge-graph source.

SNOMED CT is distributed under license, so its RF2 release is never version
controlled. This exporter runs only when a release directory is available,
located the same way as the temporal-ordering event grounder: via the
``SNOMED_DATA_PATH`` environment variable or the ``grounder.snomed_data_path``
field of the temporal-ordering config (see
:mod:`coda.grounding.temporal_ordering.event_grounding.config`).

The exported concept set is kept identical to the grounder's: active concepts
whose FSN semantic tag is one of :data:`~coda.snomed.rf2.RELEVANT_CATEGORIES`
(disorder / procedure / finding).

Every grounding surface form becomes its own node so the Neo4j vector index
covers FSNs *and* synonyms:
  - one ``status="name"`` node per concept (id ``snomedct:<conceptId>``), which
    also anchors the ``is_a`` hierarchy;
  - one ``status="synonym"`` node per accepted synonym (id
    ``snomedct:<descriptionId>``), linked to its concept with a ``synonym_of``
    edge.
Every node carries a SapBERT embedding of its own surface form and a
``concept_id`` property so a synonym hit collapses to its concept without a
graph hop.
"""
from pathlib import Path

import pandas as pd

from coda.grounding.temporal_ordering.event_grounding.config import (
    TemporalOrderingGroundingConfig,
)
from coda.kg.sources import KGSourceExporter, write_tsv_gz
from coda.kg.sources.snomedct_embeddings import (
    generate_snomedct_embeddings,
    get_node_index,
)
from coda.snomed.rf2 import iter_surface_forms, load_isa_relationships

NODE_COLUMNS = [
    "id:ID",
    ":LABEL",
    "name",
    "text",
    "category",
    "code",
    "status",
    "concept_id",
    "embedding:float[]",
]


def resolve_snomed_root() -> Path | None:
    """Return the configured SNOMED CT release directory, or None if unset."""
    return TemporalOrderingGroundingConfig.default().snomed_data_path


class SnomedCtExporter(KGSourceExporter):
    name = "snomedct"

    def __init__(self, snomed_root: Path | None = None):
        super().__init__()
        self.snomed_root = Path(snomed_root) if snomed_root else resolve_snomed_root()

    def export(self):
        if self.snomed_root is None:
            raise FileNotFoundError(
                "No SNOMED CT data directory configured; set SNOMED_DATA_PATH or "
                "grounder.snomed_data_path in the temporal-ordering config."
            )

        terminology_dir = self.snomed_root / "Snapshot" / "Terminology"

        # One node record per surface form. The FSN node is keyed by concept id
        # (and anchors the hierarchy); each synonym node is keyed by its own
        # description id and links back to the concept.
        records: list[dict] = []
        exported_concepts: set[str] = set()
        edges: list[list[str]] = []
        for sf in iter_surface_forms(self.snomed_root):
            concept_curie = f"snomedct:{sf.concept_id}"
            if sf.status == "name":
                node_id = concept_curie
                exported_concepts.add(sf.concept_id)
            else:
                node_id = f"snomedct:{sf.description_id}"
                edges.append([node_id, concept_curie, "synonym_of"])

            records.append({
                "id:ID": node_id,
                ":LABEL": "snomedct",
                "name": sf.entry_name,
                "text": sf.text,
                "category": sf.category,
                "code": sf.concept_id,
                "status": sf.status,
                "concept_id": concept_curie,
            })

        # is_a hierarchy between concept (main-term) nodes only.
        for source, destination in load_isa_relationships(
            terminology_dir, exported_concepts
        ):
            edges.append([f"snomedct:{source}", f"snomedct:{destination}", "is_a"])

        # Embed each surface form (cached across builds) and attach the vector.
        node_texts = {rec["id:ID"]: rec["text"] for rec in records}
        embeddings, surface_forms = generate_snomedct_embeddings(node_texts)
        node_to_idx = get_node_index(surface_forms)["node_to_idx"]
        for rec in records:
            vector = embeddings[node_to_idx[rec["id:ID"]]]
            rec["embedding:float[]"] = ";".join(vector.astype(str).tolist())

        nodes_df = pd.DataFrame(records, columns=NODE_COLUMNS)
        write_tsv_gz(
            nodes_df.drop_duplicates("id:ID").sort_values("id:ID"), self.nodes_file
        )

        edges_df = pd.DataFrame(edges, columns=[":START_ID", ":END_ID", ":TYPE"])
        write_tsv_gz(
            edges_df.drop_duplicates().sort_values([":START_ID", ":END_ID"]),
            self.edges_file,
        )


if __name__ == "__main__":
    exporter = SnomedCtExporter()
    exporter.export()
