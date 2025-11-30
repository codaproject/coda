from .. import CODA_BASE
from .sources import icd10, icd11, phmrc, who_va
from .io import networkx_to_tsv


KG_BASE = CODA_BASE.module('kg')


def dump_kg():
    """Dump the knowledge graph to file."""
    g = icd10.get_icd10_coda_graph()
    networkx_to_tsv(g, KG_BASE.join(name='icd10_nodes.tsv'),
                    KG_BASE.join(name='icd10_edges.tsv'))
    g = who_va.get_who_va_graph()
    networkx_to_tsv(g, KG_BASE.join(name='who_va_nodes.tsv'),
                    KG_BASE.join(name='who_va_edges.tsv'))
    g = phmrc.get_phmrc_graph()
    networkx_to_tsv(g, KG_BASE.join(name='phmrc_nodes.tsv'),
                    KG_BASE.join(name='phmrc_edges.tsv'))
    g = icd11.get_icd11_graph()
    networkx_to_tsv(g, KG_BASE.join(name='icd11_nodes.tsv'),
                    KG_BASE.join(name='icd11_edges.tsv'))


if __name__ == '__main__':
    dump_kg()
