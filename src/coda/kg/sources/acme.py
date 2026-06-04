import pandas as pd

from openacme.acme import get_acme_graph
from coda.kg.sources import KGSourceExporter, write_tsv_gz


class ACMEExporter(KGSourceExporter):
    name = "acme"

    def export(self):
        g = get_acme_graph()
        # We need to make sure all nodes have an `icd10:` prefix
        # in their label
        nodes = []
        edges = []
        for node, data in g.nodes(data=True):
            nodes.append(
                [
                    f"icd10:{node}",  # id:ID
                    "icd10",  # :LABEL
                    data.get("kind"),  # class_kind
                    node,  # code
                ]
            )
        nodes_df = pd.DataFrame(
            nodes,
            columns=["id:ID", ":LABEL", "class_kind", "code"],
        )
        write_tsv_gz(nodes_df.sort_values("id:ID"), self.nodes_file)

        for source, target, data in g.edges(data=True):
            edges.append(
                [
                    f"icd10:{source}",  # :START_ID
                    f"icd10:{target}",  # :END_ID
                    data.get("kind", "related_to"),  # :TYPE
                ]
            )
        edges_df = pd.DataFrame(edges, columns=[":START_ID", ":END_ID", ":TYPE"])
        write_tsv_gz(
            edges_df.sort_values([":START_ID", ":END_ID"]), self.edges_file
        )


if __name__ == "__main__":
    exporter = ACMEExporter()
    exporter.export()
