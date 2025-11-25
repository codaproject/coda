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
import networkx as nx
from openacme.icd10 import expand_icd10_range, get_icd10_graph
from coda.resources import get_resource_path

# Example rows
# who_va_id,who_va_name,icd10_codes
# "VAs-01","Infectious and parasitic diseases",""
# "VAs-01.01","Sepsis","A40-A41"
# "VAs-01.02","Acute respiratory infection, including pneumonia","J00-J22; J85"
# "VAs-01.03","HIV/AIDS related death","B20-B24"

WHO_VA_ICD10_MAPPINGS = get_resource_path('who.va_icd10_mappings.csv')


def get_who_va_graph():
    df = pd.read_csv(WHO_VA_ICD10_MAPPINGS)
    icd10_graph = get_icd10_graph()

    nodes = []
    edges = []
    for _, row in df.iterrows():
        who_va_id = row['who_va_id']
        who_va_curie = f'who.va:{who_va_id}'
        who_va_name = row['who_va_name']
        icd10_codes = row['icd10_codes']

        nodes.append([
            who_va_curie, {
                'name': who_va_name
            }
        ])

        # Determine parent based on ID structure
        if '.' in who_va_id:
            parent_id = who_va_id.rsplit('.', 1)[0]
            parent_curie = f'who.va:{parent_id}'
            edges.append((who_va_curie, parent_curie, {'type': 'isa'}))

        # Parse ICD codes
        if pd.notna(icd10_codes) and icd10_codes.strip():
            for code_part in icd10_codes.split(';'):
                code_part = code_part.strip()
                if '-' in code_part:
                    start_code, end_code = code_part.split('-', 1)
                    codes = expand_icd10_range(icd10_graph,
                                               start_code.strip(),
                                               end_code.strip())
                else:
                    codes = [code_part]

                for code in codes:
                    icd10_curie = f'icd10:{code}'
                    edges.append((
                        icd10_curie,
                        who_va_curie,
                        {'type': 'maps_to'}
                    ))
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


if __name__ == '__main__':
    g = get_who_va_graph()
    print(f'WHO VA graph has {len(g.nodes)} nodes and {len(g.edges)} edges.')