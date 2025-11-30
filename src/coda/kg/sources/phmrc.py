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
import pandas as pd
import networkx as nx

from coda.resources import get_resource_path

PHMRC_ICD10_MAPPINGS = get_resource_path('phmrc_icd10_mappings.csv')


def process_phmrc_icd10_mappings(phmrc_path):
    """Parse PHMRC data file to extract mappings to ICD codes.

    Parameters
    ----------
    phmrc_path : str
        Path to the PHMRC CSV data file, e.g.,
        IHME_PHMRC_VA_DATA_ADULT_Y2013M09D11_0.csv which requires
        registration to download, see module documentation.
    """
    df = pd.read_csv(phmrc_path)

    mappings = set()
    for _, row in df.iterrows():
        phmrc_name = row['gs_text55']
        icd10_code = row['gs_code55']

        mappings.add((phmrc_name, icd10_code))

    mappings = sorted(mappings, key=lambda x: x[0])
    out_df = pd.DataFrame(mappings, columns=['phmrc_name', 'icd10_code'])
    out_df.to_csv(PHMRC_ICD10_MAPPINGS, index=False)


def get_phmrc_graph():
    df = pd.read_csv(PHMRC_ICD10_MAPPINGS)
    nodes = []
    edges = []
    for _, row in df.iterrows():
        phmrc_name = row['phmrc_name']
        icd10_code = row['icd10_code']

        phmrc_curie = f'phmrc:{phmrc_name}'

        nodes.append([
            phmrc_curie, {
                'name': phmrc_name
            }
        ])

        if pd.notna(icd10_code) and icd10_code.strip():
            edges.append((
                f'icd10:{icd10_code}',
                phmrc_curie,
                {'kind': 'maps_to'}
            ))
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


if __name__ == '__main__':
    g = get_phmrc_graph()
