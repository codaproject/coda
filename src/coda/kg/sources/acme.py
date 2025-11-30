import networkx as nx

from openacme.acme import get_acme_graph


def get_acme_coda_graph():
    g = get_acme_graph()
    # We need to make sure all nodes have an `icd10:` prefix
    # in their label
    mapping = {}
    for node, data in g.nodes(data=True):
        mapping[node] = f'icd10:{node}'
        name = data.get('rubrics', {}).pop('preferred', [None])[0]
        if name:
            g.nodes[node]['name'] = name
    g = nx.relabel_nodes(g, mapping)
    # Next we need to unpack the rubtics to get the "preferred" name
    for node in g.nodes:
        code = node.replace('icd10:', '')
        g.nodes[node]['code'] = code
    return g
