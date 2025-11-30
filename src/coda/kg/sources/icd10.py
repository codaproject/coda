import networkx as nx

from openacme.icd10 import ICD10_BASE, ICD10_XML_URL, get_icd10_graph, \
    expand_icd10_range


def get_icd10_coda_graph():
    g = get_icd10_graph()
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
