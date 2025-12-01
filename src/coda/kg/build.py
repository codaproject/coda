from pathlib import Path

from .sources import icd10, icd11, phmrc, who_va, acme, probbase
from .io import networkx_to_tsv


HERE = Path(__file__).parent
KG_BASE = HERE.parent.parent.parent.joinpath('kg')

def dump_kg():
    """Dump the knowledge graph to file."""
    # Make folder if needed
    KG_BASE.mkdir(exist_ok=True)
    g = icd10.get_icd10_coda_graph()
    networkx_to_tsv(g, KG_BASE / 'icd10_nodes.tsv',
                    KG_BASE / 'icd10_edges.tsv')
    g = acme.get_acme_coda_graph()
    networkx_to_tsv(g, KG_BASE / 'acme_nodes.tsv',
                    KG_BASE / 'acme_edges.tsv')
    g = who_va.get_who_va_graph()
    networkx_to_tsv(g, KG_BASE / 'who_va_nodes.tsv',
                    KG_BASE / 'who_va_edges.tsv')
    g = phmrc.get_phmrc_graph()
    networkx_to_tsv(g, KG_BASE / 'phmrc_nodes.tsv',
                    KG_BASE / 'phmrc_edges.tsv')
    g = icd11.get_icd11_graph()
    networkx_to_tsv(g, KG_BASE / 'icd11_nodes.tsv',
                    KG_BASE / 'icd11_edges.tsv')
    g = probbase.get_probbase_graph()
    networkx_to_tsv(g, KG_BASE / 'probbase_nodes.tsv',
                    KG_BASE / 'probbase_edges.tsv')


if __name__ == '__main__':
    dump_kg()
