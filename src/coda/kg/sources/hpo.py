import pandas as pd
import networkx as nx
import obonet

from coda import CODA_BASE

HPO_BASE = CODA_BASE.module('hpo')
HPOA_URL = "https://purl.obolibrary.org/obo/hp/phenotype.hpoa"
HPO_URL = "https://purl.obolibrary.org/obo/hp.obo"

# database_id»disease_name»···qualifier»··hpo_id»·reference»··evidence»···onset»··frequency»··sex»modifier»···aspect»·biocuration
#      6 OMIM:619340»Developmental and epileptic encephalopathy 96»··»···HP:0011097»·PMID:31675180»··PCS»»···1/2»»···»···P»··HPO:probinson[2021-06-21]
#      7 OMIM:619340»Developmental and epileptic encephalopathy 96»··»···HP:0002187»·PMID:31675180»··PCS»»···1/1»»···»···P»··HPO:probinson[2021-06-21]

def get_hpoa_graph():
    hpoa_file = HPO_BASE.ensure(url=HPOA_URL)
    hpo_file = HPO_BASE.ensure(url=HPO_URL)
    df = pd.read_csv(hpoa_file, sep='\t', skiprows=4)
    hp_graph = obonet.read_obo(hpo_file)
    nodes = []
    edges = []
    disease_added = set()
    phenotypes_added = set()
    for _, row in df.iterrows():
        disease_id = row['database_id']
        disease_name = row['disease_name']
        hpo_id = row['hpo_id']

        # FIXME: check if this is actually always valid
        disease_curie = disease_id.lower()
        phenotype_curie = hpo_id.lower()

        if disease_curie not in disease_added:
            disease_ns = disease_curie.split(':')[0]
            nodes.append([
                disease_curie, {
                    'name': disease_name,
                    'kind': disease_ns
                }
            ])
            disease_added.add(disease_curie)

        if phenotype_curie not in phenotypes_added:
            hp_term_name = hp_graph.nodes[hpo_id]['name']
            nodes.append([
                phenotype_curie, {
                    'kind': 'hp',
                    'name': hp_term_name,
                }
            ])
            phenotypes_added.add(phenotype_curie)

        edges.append((
            disease_curie,
            phenotype_curie,
            {
                'evidence': row['evidence'],
                'frequency': row['frequency'],
                'onset': row['onset'],
                'kind': 'has_phenotype',
                'qualifier': row['qualifier'],
                'sex': row['sex'],
                'aspect': row['aspect'],
                'modifier': row['modifier'],
            }
        ))
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g
