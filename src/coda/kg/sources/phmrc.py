"""
This module processes custom terms used by PHMRC (Public Health
Medical Research Consortium) in their verbal autopsy data
collection and links them to standard ontologies such as
ICD-10 codes.

The data files for PHMRC can be accessed at
https://ghdx.healthdata.org/record/ihme-data/population-health-metrics-research-consortium-gold-standard-verbal-autopsy-data-2005-2011
and are only downloadable after registration.

IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv
"""
import logging

import pandas as pd
from coda.kg.sources import KGSourceExporter
from coda.resources import get_resource_path


logger = logging.getLogger(__name__)


PHMRC_RAW_DATA = get_resource_path("IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv")
PHMRC_ICD10_MAPPINGS = get_resource_path("phmrc_icd10_mappings.csv")


def process_phmrc_icd10_mappings(phmrc_path: str = PHMRC_RAW_DATA):
    """Parse PHMRC data file to extract mappings to ICD codes.

    Parameters
    ----------
    phmrc_path :
        Path to the PHMRC CSV data file, e.g.,
        IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv which requires
        registration to download, see module documentation.
    """
    try:
        df = pd.read_csv(phmrc_path)
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"Could not find {phmrc_path} PHMRC data file. Please download it "
            f"from https://ghdx.healthdata.org/record/ihme-data"
            "/population-health-metrics-research-consortium-gold-standard-"
            "verbal-autopsy-data-2005-2011 after registration. (See module "
            "documentation of {__file__} for details.)"
        ) from err

    mappings = set()
    for _, row in df.iterrows():
        phmrc_name = row["gs_text55"]
        icd10_code = row["gs_code55"]

        mappings.add((phmrc_name, icd10_code))

    mappings = sorted(mappings, key=lambda x: x[0])
    out_df = pd.DataFrame(mappings, columns=["phmrc_name", "icd10_code"])
    out_df.to_csv(PHMRC_ICD10_MAPPINGS, index=False)


class PhmrcExporter(KGSourceExporter):
    name = "phmrc"

    def export(self):
        # phmrc nodes:
        # - id:ID "phmrc:<phmrc_name>"
        # - name: <phmrc_name>
        # - :LABEL "phmrc"
        # edges:
        # - icd10 curie -[ :maps_to ]-> phmrc curie
        process_phmrc_icd10_mappings()

        df = pd.read_csv(PHMRC_ICD10_MAPPINGS)

        df["phmrc_curie"] = df["phmrc_name"].apply(lambda x: f"phmrc:{x}")
        df["icd10_curie"] = df["icd10_code"].apply(
            lambda x: f"icd10:{x}" if pd.notna(x) and x.strip() else None
        )

        # Dump all phmrc entities as nodes with :LABEL "phmrc"
        phmrc_nodes = df[["phmrc_curie", "phmrc_name"]].drop_duplicates()
        phmrc_nodes = phmrc_nodes.rename(
            columns={"phmrc_curie": "id:ID", "phmrc_name": "name"}
        )
        phmrc_nodes[":LABEL"] = "phmrc"
        phmrc_nodes.sort_values("id:ID").to_csv(
            self.nodes_file, sep="\t", index=False
        )

        # Dump the mappings as edges
        edges = df[pd.notna(df["icd10_curie"])][["icd10_curie", "phmrc_curie"]]
        edges = edges.rename(
            columns={"icd10_curie": ":START_ID", "phmrc_curie": ":END_ID"}
        )
        edges[":TYPE"] = "maps_to"
        edges.sort_values([":START_ID", ":END_ID"]).to_csv(
            self.edges_file, sep="\t", index=False
        )


if __name__ == "__main__":
    exporter = PhmrcExporter()
    exporter.export()
