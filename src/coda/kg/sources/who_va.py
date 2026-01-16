"""
This module processes the WHO Verbal Autopsy (VA) classifications
into terms and builds a graph representation along with mappings
to ICD-10 codes / ranges.

The WHO VA classifications are available at:
https://www.who.int/publications/m/item/verbal-autopsy-standards-the-2016-who-verbal-autopsy-instrument
in particular, the Download link contains a file called
01_Manual and guidelines for application and use of simplified WHO VA tool_2016 _V1.5.3.pdf
which contains an Appendix 1 listing the VA cause categories and their
corresponding ICD-10 codes / ranges.

In addition, supplement to the paper
https://pmc.ncbi.nlm.nih.gov/articles/PMC3433652/ also contains a table
which is somwhat different from the WHO listing, potentially
due to updates (the paper is from 2012 whereas the latest WHO VA
standard is from 2016).
"""
import pandas as pd
from coda.kg.sources import KGSourceExporter
from openacme.icd10 import expand_icd10_range, get_icd10_graph
from coda.resources import get_resource_path

# Example rows
# who_va_id,who_va_name,icd10_codes
# "VAs-01","Infectious and parasitic diseases",""
# "VAs-01.01","Sepsis","A40-A41"
# "VAs-01.02","Acute respiratory infection, including pneumonia","J00-J22; J85"
# "VAs-01.03","HIV/AIDS related death","B20-B24"

WHO_VA_ICD10_MAPPINGS = get_resource_path("who.va_icd10_mappings.csv")


class WhoVaExporter(KGSourceExporter):
    name = "who_va"

    def export(self):
        df = pd.read_csv(WHO_VA_ICD10_MAPPINGS)

        # Graph is used in expanding ICD-10 ranges
        icd10_graph = get_icd10_graph()

        # Set WHO VA curies
        df["who_va_curie"] = df["who_va_id"].apply(lambda x: f"who.va:{x}")

        # Dump the who.va nodes
        nodes = df[["who_va_curie", "who_va_name"]].rename(
            {
                "who_va_curie": "id:ID",
                "who_va_name": "name",
            },
            axis=1,
        )
        nodes[":LABEL"] = "who.va"
        nodes.sort_values("id:ID").to_csv(self.nodes_file, sep="\t", index=False)

        edges = []
        for _, row in df.iterrows():
            who_va_id = row["who_va_id"]
            who_va_curie = row["who_va_curie"]
            icd10_codes = row["icd10_codes"]

            # Determine parent based on ID structure
            if "." in who_va_id:
                parent_id = who_va_id.rsplit(".", 1)[0]
                parent_curie = f"who.va:{parent_id}"
                edges.append((who_va_curie, parent_curie, "is_a"))

            # Parse ICD codes
            if pd.notna(icd10_codes) and icd10_codes.strip():
                for code_part in icd10_codes.split(";"):
                    code_part = code_part.strip()
                    if "-" in code_part:
                        start_code, end_code = code_part.split("-", 1)
                        codes = expand_icd10_range(
                            icd10_graph, start_code.strip(), end_code.strip()
                        )
                    else:
                        codes = [code_part]

                    for code in codes:
                        icd10_curie = f"icd10:{code}"
                        edges.append((icd10_curie, who_va_curie, "maps_to"))

        edges_df = pd.DataFrame(edges, columns=[":START_ID", ":END_ID", ":TYPE"])
        edges_df.sort_values([":START_ID", ":END_ID"]).to_csv(
            self.edges_file, sep="\t", index=False
        )


if __name__ == "__main__":
    who_va_exporter = WhoVaExporter()
    who_va_exporter.export()
