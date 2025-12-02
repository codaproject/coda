"""
The ICD11 spreadsheet can be downloaded from
https://icdcdn.who.int/static/releasefiles/2025-01/SimpleTabulation-ICD-11-MMS-en.zip
This zip file has both a simple tsv file with a txt extension and an XLSX file.
The columns are as follows:
Foundation URI	Linearization URI	Code	BlockId	Title	ClassKind	DepthInKind
IsResidual	ChapterNo	BrowserLink	isLeaf	Primary tabulation
Grouping1	Grouping2	Grouping3	Grouping4	Grouping5
Version:2025 Jan 24 - 22:30 UTC
"""
import zipfile

import networkx as nx
import pandas as pd

from openacme import OPENACME_BASE

ICD11_BASE = OPENACME_BASE.module('icd11')
ICD11_ZIP_URL = "https://icdcdn.who.int/static/releasefiles/2025-01/SimpleTabulation-ICD-11-MMS-en.zip"
ICD11_FNAME = "SimpleTabulation-ICD-11-MMS-en.txt"
ICD11_MAPPINGS_URL = "https://icdcdn.who.int/static/releasefiles/2025-01/mapping.zip"
ICD11_MAPPINGS_FNAME = "foundation_11To10MapToOneCategory.xlsx"


def get_icd11_graph():
    zip_path = ICD11_BASE.ensure(url=ICD11_ZIP_URL)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(ICD11_FNAME) as fh:
            df = pd.read_csv(fh, sep='\t')

    # Foundation URI	icd11Code	icd11Chapter	icd11Title	icd10Code	icd10Title
    # http://id.who.int/icd/entity/1435254666	01	01	Certain infectious or parasitic diseases	I	Certain infectious and parasitic diseases
    map_zip_path  = ICD11_BASE.ensure(url=ICD11_MAPPINGS_URL)
    with zipfile.ZipFile(map_zip_path, 'r') as zf:
        with zf.open(ICD11_MAPPINGS_FNAME) as fh:
            mapping_df = pd.read_excel(fh, sheet_name='foundation_11To10MapToOneCateg')
    icd11_to_10 = {}
    for _, row in mapping_df.iterrows():
        foundation_id = row['Foundation URI'].split('/')[-1]
        foundation_curie = f'icd11:{foundation_id}'
        icd10_code = row['icd10Code']
        icd10_curie = f'icd10:{icd10_code}'
        icd11_to_10[foundation_curie] = icd10_curie

    nodes = []
    edges = []
    prev_depth = 1
    parent_depths = {1: None, 2: None, 3: None}
    for _, row in df.iterrows():
        if not row['IsResidual']:
            foundation_id = row['Foundation URI'].split('/')[-1]
            foundation_curie = f'icd11:{foundation_id}'
        # TODO: handle residual categories
        # http://id.who.int/icd/release/11/mms/344162786	1A03
        #   - - - Intestinal infections due to Escherichia coli
        # http://id.who.int/icd/release/11/mms/344162786/other	1A03.Y
        #   - - - - Intestinal infections due to other specified Escherichia coli
        # http://id.who.int/icd/release/11/mms/344162786/unspecified	1A03.Z
        #   - - - - Intestinal infections due to Escherichia coli, unspecified
        else:
            continue

        if not pd.isna(row['Code']):
            icd11_code = row['Code']
        else:
            icd11_code = None
        
        title = row['Title'].replace('- ', '')
        nodes.append([
            foundation_curie, {
                'code': icd11_code,
                'name': title,
                'class_kind': row['ClassKind'],
                'kind': 'icd11',
            }
        ])
        # FIXME: this logic needs to be updated to handle
        # the concept of "DepthInKind" properly
        depth = int(row['DepthInKind'])
        if depth > 1:
            parent = parent_depths[depth-1]
        else:
            parent = None
        parent_depths[depth] = foundation_id
        if parent is not None:
            parent_curie = f'icd11:{parent}'
            edges.append((foundation_curie, parent_curie, {'kind': 'is_a'}))

        if foundation_curie in icd11_to_10:
            icd10_curie = icd11_to_10[foundation_curie]
            nodes.append([icd10_curie,
                          {'redundant': True}])
            edges.append((
                foundation_curie,
                icd10_curie,
                {'kind': 'maps_to'}
            ))

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


if __name__ == '__main__':
    g = get_icd11_graph()