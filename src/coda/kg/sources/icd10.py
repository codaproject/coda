import pandas as pd
from pathlib import Path

from coda.kg.sources import KGSourceExporter
from openacme.icd10 import get_icd10_graph
from openacme.icd10.generate_embeddings import (generate_icd10_embeddings, 
                                                load_embeddings, get_code_index)
from openacme import OPENACME_BASE

ICD10_EMBEDDINGS_BASE = OPENACME_BASE.module("icd10_embeddings")


class ICD10Exporter(KGSourceExporter):
    name = "icd10"

    def export(self):
        g = get_icd10_graph()
        # Load associated ICD10 embeddings
        if not (Path(ICD10_EMBEDDINGS_BASE.base) / "embeddings.npy").exists():
            _ = generate_icd10_embeddings()
        icd10_embeddings, definitions_data = load_embeddings()

        icd10_to_embedding_map = get_code_index(definitions_data=definitions_data)[
            "code_to_idx"
        ]
        # We need to make sure all nodes have an `icd10:` prefix
        # in their label
        nodes = []
        edges = []
        for node, data in g.nodes(data=True):
            # Find associated embedding and format for writing to tsv
            node_idx = icd10_to_embedding_map.get(node)
            embedding = (
                ";".join(icd10_embeddings[node_idx].astype(str).tolist())
                if node_idx is not None
                else ""
            )

            nodes.append(
                [
                    f"icd10:{node}",  # id:ID
                    "icd10",  # kind -> :LABEL
                    data.get("rubrics", {}),  # rubrics
                    data.get("rubrics", {}).pop("preferred", [None])[0],  # name
                    data.get("kind"),  # class_kind
                    node,  # code
                    embedding,  # associated embedding
                ]
            )
        nodes_df = pd.DataFrame(
            nodes,
            columns=[
                "id:ID",
                ":LABEL",
                "rubrics",
                "name",
                "class_kind",
                "code",
                "embedding:float[]",
            ],
        )
        nodes_df.sort_values("id:ID").to_csv(self.nodes_file, sep="\t", index=False)

        for source, target, data in g.edges(data=True):
            edges.append(
                [
                    f"icd10:{source}",  # :START_ID
                    f"icd10:{target}",  # :END_ID
                    data.get("kind", "related_to"),  # :TYPE
                ]
            )
        edges_df = pd.DataFrame(edges, columns=[":START_ID", ":END_ID", ":TYPE"])
        edges_df.sort_values([":START_ID", ":END_ID"]).to_csv(
            self.edges_file, sep="\t", index=False
        )


if __name__ == "__main__":
    exporter = ICD10Exporter()
    exporter.export()
