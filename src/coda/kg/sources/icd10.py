import networkx as nx

from coda.kg.io import networkx_to_tsv
from coda.kg.sources import KGSourceExporter
from openacme.icd10 import get_icd10_graph


def get_icd10_coda_graph():
    g = get_icd10_graph()
    # We need to make sure all nodes have an `icd10:` prefix
    # in their label
    mapping = {}
    for node, data in g.nodes(data=True):
        mapping[node] = f"icd10:{node}"
        name = data.get("rubrics", {}).pop("preferred", [None])[0]
        if name:
            g.nodes[node]["name"] = name
        g.nodes[node]["class_kind"] = data.get("kind")
        g.nodes[node]["kind"] = "icd10"
    g = nx.relabel_nodes(g, mapping)
    # Next we need to unpack the rubtics to get the "preferred" name
    for node in g.nodes:
        code = node.replace("icd10:", "")
        g.nodes[node]["code"] = code
    return g


class ICD10Exporter(KGSourceExporter):
    name = "icd10"

    def export(self):
        g = get_icd10_coda_graph()
        networkx_to_tsv(g, self.nodes_file, self.edges_file)


if __name__ == "__main__":
    exporter = ICD10Exporter()
    exporter.export()
