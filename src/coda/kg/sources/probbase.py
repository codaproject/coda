"""
This module gets the list of VA questions from the "probbase" associated
with InterVA and construct nodes for the questions themselves, and
build edges between the questions and the causes they relate to
using the information in the probbase.
"""

__all__ = ["get_probbase_graph"]

import pandas as pd
import networkx as nx


PROBBASE_URL = ("https://github.com/verbal-autopsy-software/interva/raw/"
                "refs/heads/main/src/interva/data/probbase.xls")

def process_va_col(col_name):
    assert col_name.startswith('b_')
    code = col_name[2:]
    if code.endswith('00'):
        code = code[:-2]
    else:
        code = f'{code[:2]}.{code[2:]}'
    return f'who.va:{code}'


def get_probbase_graph():
    df = pd.read_excel(PROBBASE_URL, sheet_name='probbase')
    id_column = 'who_2016'
    name_column = 'qdesc'
    prop_columns = ['indic', 'sdesc', 'ilab', 'subst', 'samb']
    va_question_cols = {
        col: process_va_col(col) for col in df.columns if col.startswith('b_')
    }
    nodes = []
    edges = []
    for _, row in df.iterrows():
        if pd.isna(row['indic']):
            continue
        node_curie = f'who.va.q:{row[id_column]}'
        node = [
            node_curie, {
                'name': row[name_column],
                **{prop: row[prop] for prop in prop_columns}
            }
        ]
        nodes.append(node)
        for col, va_curie in va_question_cols.items():
            edge = [
                node_curie,
                va_curie,
                {
                    'kind': 'probbase_rel',
                    'value': row[col]
                }
            ]
            edges.append(edge)
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g
