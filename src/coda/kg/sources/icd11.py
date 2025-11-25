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


def get_icd11_graph():
    zip_path = ICD11_BASE.ensure(url=ICD11_ZIP_URL)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(ICD11_FNAME) as fh:
            df = pd.read_csv(fh, sep='\t')

    nodes = []
    edges = []
    prev_depth = 1
    parent_depths = {1: None, 2: None, 3: None}
    for _, row in df.iterrows():
        if not row['IsResidual']:
            foundation_id = row['Foundation URI'].split('/')[-1]
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
            foundation_id, {
                'icd11_code': icd11_code,
                'name': title,
                'kind': row['ClassKind'],
            }
        ])

        depth = int(row['DepthInKind'])
        if depth > 1:
            parent = parent_depths[depth-1]
        else:
            parent = None
        parent_depths[depth] = foundation_id
        if parent is not None:
            edges.append((foundation_id, parent, {'kind': 'is_a'}))

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


if __name__ == '__main__':
    g = get_icd11_graph()