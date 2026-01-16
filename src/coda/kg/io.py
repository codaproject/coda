__all__ = ['networkx_to_tsv']

import pandas as pd


def networkx_to_tsv(g, node_path, edge_path):
    """Export a NetworkX graph to TSV files for nodes and edges

    The node and edge files are compatible with neo4j.

    Parameters
    ----------
    g : networkx.Graph
        The input graph.
    node_path : str | Path
        Path to output TSV file for nodes.
    edge_path : str | Path
        Path to output TSV file for edges.
    """
    # Export nodes
    node_rows = []
    for node, data in g.nodes(data=True):
        if data.get('redundant'):
            continue
        row = {'id:ID': node, ':LABEL': data.pop('kind', 'Entity')}
        row.update(data)
        node_rows.append(row)
    node_df = pd.DataFrame(node_rows)
    node_df.to_csv(node_path, sep='\t', index=False)

    # Export edges
    edge_rows = []
    for source, target, data in g.edges(data=True):
        row = {':START_ID': source, ':END_ID': target,
               ':TYPE': data.pop('kind', 'related_to')}
        row.update(data)
        edge_rows.append(row)
    edge_df = pd.DataFrame(edge_rows)
    edge_df.to_csv(edge_path, sep='\t', index=False)
