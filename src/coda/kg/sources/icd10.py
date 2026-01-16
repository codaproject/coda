import pandas as pd

from coda.kg.sources import KGSourceExporter
from openacme.icd10 import get_icd10_graph


class ICD10Exporter(KGSourceExporter):
    name = "icd10"

    def export(self):
        g = get_icd10_graph()
        # We need to make sure all nodes have an `icd10:` prefix
        # in their label
        nodes = []
        edges = []
        for node, data in g.nodes(data=True):
            nodes.append(
                [
                    f"icd10:{node}",  # id:ID
                    "icd10",  # kind -> :LABEL
                    data.get("rubrics", {}),  # rubrics
                    data.get("rubrics", {}).pop("preferred", [None])[0],  # name
                    data.get("kind"),  # class_kind
                    node,  # code
                ]
            )
        nodes_df = pd.DataFrame(
            nodes,
            columns=["id:ID", ":LABEL", "rubrics", "name", "class_kind", "code"],
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
