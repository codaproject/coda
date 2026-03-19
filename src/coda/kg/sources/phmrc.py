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
from collections import defaultdict

import pandas as pd
from coda.kg.sources import KGSourceExporter
from coda.resources import get_resource_path


logger = logging.getLogger(__name__)


PHMRC_COD_TERMS = get_resource_path("smart_va_cause_list.csv")


def _normalize_name(name):
    # Remove parentheticals, e.g., Homicide (assault) -> Homicide
    return name.split(" (")[0].strip()


class PhmrcExporter(KGSourceExporter):
    name = "phmrc"

    def export(self):
        # phmrc nodes:
        # - id:ID "phmrc:<phmrc_name>"
        # - name: <phmrc_name>
        # - :LABEL "phmrc"
        # edges:
        # - icd10 curie -[ :maps_to ]-> phmrc curie

        df = pd.read_csv(PHMRC_COD_TERMS)
        # Get a list of smart_va code to name mappings
        # Handle corner cases like D91 (G96)
        # Add age_groups as an attribute
        # Add cod_groups as an attribute
        # Add note as an attibute
        # Add mappings edge
        phmrc_code_to_term = defaultdict(list)
        for _, row in df.iterrows():
            # Here both codes are relevant and can be mapped
            if row['smart_va'] == 'D91 (G96)':
                codes = ['D91', 'G96']
                names = ['Leukemia/Lymphomas', 'Lymphomas']
            else:
                codes = [row['smart_va']]
                names = [_normalize_name(row['name'])]
            for code, name in zip(codes, names):
                # We simplify this one
                if row['icd10'] == 'P23/J22':
                    icd10_code = 'P23'
                else:
                    icd10_code = row['icd10']
                data = {
                    'age_group': row['age_group'],
                    'cod_group': row['group'] if not pd.isna(row['group']) else 'None',
                    'name': name,
                    'comments': row['comments'],
                    'icd10_code': icd10_code,
                }
                phmrc_code_to_term[code].append(data)

        # Join the data terms using |-separated strings
        merged_phmrc_records = []
        edge_records = []
        for code, data_list in phmrc_code_to_term.items():
            merged_data = {
                'id:ID': f"phmrc:{code}",
                ':LABEL': 'phmrc',
                'age_group': '|'.join(set(d['age_group'] for d in data_list)),
                'cod_group': '|'.join(set(d['cod_group'] for d in data_list)),
                'name': '|'.join(set(d['name'] for d in data_list)),
                'comments': '|'.join(set(d['comments'] for d in data_list)),
            }
            merged_phmrc_records.append(merged_data)
            edge_records.append(
                {':START_ID': f'phmrc:{code}',
                 ':END_ID': f"icd10:{data_list[0]['icd10_code']}",
                 ':TYPE': 'maps_to'}
            )

        phmrc_nodes = pd.DataFrame(merged_phmrc_records)
        edges = pd.DataFrame(edge_records)

        phmrc_nodes.sort_values("id:ID").to_csv(
            self.nodes_file, sep="\t", index=False
        )

        # Dump the mappings as edges
        edges.sort_values([":START_ID", ":END_ID"]).to_csv(
            self.edges_file, sep="\t", index=False
        )


if __name__ == "__main__":
    exporter = PhmrcExporter()
    exporter.export()
