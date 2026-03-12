import pandas as pd
import zipfile

from coda.kg.sources import KGSourceExporter
from openacme.icd10 import get_icd10_graph
from openacme.icd10.generate_embeddings import load_embeddings, get_code_index
from openacme import OPENACME_BASE
from coda.resources import get_resource_path

ICD10_EMBEDDINGS_BASE = OPENACME_BASE.module("icd10_embeddings")
PHMRC_ICD10_MAPPINGS = get_resource_path("phmrc_icd10_mappings.csv")
ICD11_BASE = OPENACME_BASE.module("icd11")
ICD11_MAPPINGS_URL = "https://icdcdn.who.int/static/releasefiles/2025-01/mapping.zip"
ICD11_MAPPINGS_FNAME = "foundation_11To10MapToOneCategory.xlsx"


class ICD10Exporter(KGSourceExporter):
    name = "icd10"

    def export(self):
        g = get_icd10_graph()
        # Load associated ICD10 embeddings
        icd10_embeddings, definitions_data = load_embeddings(
            embeddings_base=ICD10_EMBEDDINGS_BASE
        )
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
        ## add ICD10 nodes from other sources ##
        nodes = add_missing_icd10_nodes(nodes)
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


def add_missing_icd10_nodes(nodes: list):
    """Add ICD10 nodes we expect to be missing from the ICD10 graph, using other sources"""
    primary_curies = [x[0] for x in nodes]
    ## get ICD10 nodes only present in ICD11 edges ##
    map_zip_path = ICD11_BASE.ensure(url=ICD11_MAPPINGS_URL)
    with zipfile.ZipFile(map_zip_path, "r") as zf:
        with zf.open(ICD11_MAPPINGS_FNAME) as fh:
            mapping_df = pd.read_excel(fh, sheet_name="foundation_11To10MapToOneCateg")
    for _, row in mapping_df.iterrows():
        icd10_code = row["icd10Code"]
        icd10_curie = f"icd10:{icd10_code}"
        if icd10_curie not in primary_curies:
            nodes.append(
                [
                    icd10_curie,  # id:ID
                    "icd10",  # kind -> :LABEL
                    {},  # rubrics
                    mapping_df[mapping_df["icd10Code"] == icd10_code][
                        "icd10Title"
                    ].values[
                        0
                    ],  # name
                    "missing",  # class_kind
                    icd10_code,  # code
                    ";".join([str(0) for _ in range(32)]),  # associated embedding
                ]
            )
    ## get ICD10 nodes only present in PMRC EDGES ##
    df = pd.read_csv(PHMRC_ICD10_MAPPINGS)
    for _, row in df.iterrows():
        code = row.get("icd10_code")
        curie = f"icd10:{code}"
        if curie not in primary_curies:
            nodes.append(
                [
                    curie,  # id:ID
                    "icd10",  # kind -> :LABEL
                    {},  # rubrics
                    row.get("phmrc_name"),  # name
                    "missing",  # class_kind
                    code,  # code
                    ";".join([str(0) for _ in range(32)]),  # associated embedding
                ]
            )
    return nodes


if __name__ == "__main__":
    exporter = ICD10Exporter()
    exporter.export()
