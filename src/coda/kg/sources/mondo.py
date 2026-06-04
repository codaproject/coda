import obonet
import pandas as pd

from coda import CODA_BASE
from coda.kg.sources import KGSourceExporter

MONDO_BASE = CODA_BASE.module("mondo")
MONDO_URL = "https://purl.obolibrary.org/obo/mondo.obo"

# MONDO records cross-references as skos:exactMatch entries in the
# property_value attribute, e.g. "skos:exactMatch OMIM:190685". We only
# export exact matches to the namespaces below, mapping each MONDO prefix
# to the standard namespace used in the CODA KG. This could also be done
# using the Bioregistry but this ensures local control over the standard.
NAMESPACE_MAP = {
    "ICD10WHO": "icd10",
    "icd11.foundation": "icd11",
    "OMIM": "omim",
    "MESH": "mesh",
    "Orphanet": "orpha",
}

EXACT_MATCH = "skos:exactMatch"


class MondoExporter(KGSourceExporter):
    name = "mondo"

    def export(self):
        mondo_file = MONDO_BASE.ensure(url=MONDO_URL)
        g = obonet.read_obo(mondo_file)

        nodes = []
        edges = []
        for node, data in g.nodes(data=True):
            # Only instantiate MONDO terms as nodes
            if not node.startswith("MONDO:"):
                continue

            mondo_curie = node.lower()
            nodes.append(
                [
                    mondo_curie,  # id:ID
                    "mondo",  # :LABEL
                    data.get("name"),  # name
                ]
            )

            # Traverse property_value entries and keep skos:exactMatch
            # relations to accepted namespaces
            for prop in data.get("property_value", []):
                relation, _, value = prop.partition(" ")
                if relation != EXACT_MATCH:
                    continue
                xref = value.strip()
                prefix, sep, identifier = xref.partition(":")
                if not sep or prefix not in NAMESPACE_MAP:
                    continue
                target_curie = f"{NAMESPACE_MAP[prefix]}:{identifier}"
                edges.append([mondo_curie, target_curie, EXACT_MATCH])

        nodes_df = pd.DataFrame(nodes, columns=["id:ID", ":LABEL", "name"])
        nodes_df.drop_duplicates().sort_values("id:ID").to_csv(
            self.nodes_file, sep="\t", index=False
        )

        edges_df = pd.DataFrame(edges, columns=[":START_ID", ":END_ID", ":TYPE"])
        edges_df.drop_duplicates().sort_values([":START_ID", ":END_ID"]).to_csv(
            self.edges_file, sep="\t", index=False
        )


if __name__ == "__main__":
    exporter = MondoExporter()
    exporter.export()
