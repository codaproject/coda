import pandas as pd
import obonet

from coda import CODA_BASE
from coda.kg.sources import KGSourceExporter

HPO_BASE = CODA_BASE.module("hpo")
HPOA_URL = "https://purl.obolibrary.org/obo/hp/phenotype.hpoa"
HPO_URL = "https://purl.obolibrary.org/obo/hp.obo"

# database_id»disease_name»···qualifier»··hpo_id»·reference»··evidence»···onset»··frequency»··sex»modifier»···aspect»·biocuration
#      6 OMIM:619340»Developmental and epileptic encephalopathy 96»··»···HP:0011097»·PMID:31675180»··PCS»»···1/2»»···»···P»··HPO:probinson[2021-06-21]
#      7 OMIM:619340»Developmental and epileptic encephalopathy 96»··»···HP:0002187»·PMID:31675180»··PCS»»···1/1»»···»···P»··HPO:probinson[2021-06-21]


class HpoExporter(KGSourceExporter):
    name = "hpo"

    def export(self):
        hpoa_file = HPO_BASE.ensure(url=HPOA_URL)
        hpo_file = HPO_BASE.ensure(url=HPO_URL)
        df = pd.read_csv(hpoa_file, sep="\t", skiprows=4)

        # Set curies and types
        df["disease_curie"] = df["database_id"].str.lower()
        df["phenotype_curie"] = df["hpo_id"].str.lower()
        df["disease_type"] = df["database_id"].str.split(":").str[0].str.lower()
        df["phenotype_type"] = df["hpo_id"].str.split(":").str[0].str.lower()

        # Load HPO graph to get term names
        hp_graph = obonet.read_obo(hpo_file)
        df["hpo_name"] = df["hpo_id"].map(
            lambda x: hp_graph.nodes[x]["name"] if x in hp_graph.nodes else None
        )

        # Dump disease and phenotype nodes
        pd.concat(
            [
                df[["disease_curie", "disease_type", "disease_name"]].rename(
                    columns={
                        "disease_curie": "id:ID",
                        "disease_type": ":LABEL",
                        "disease_name": "name",
                    }
                ),
                df[["phenotype_curie", "phenotype_type", "hpo_name"]].rename(
                    columns={
                        "phenotype_curie": "id:ID",
                        "phenotype_type": ":LABEL",
                        "hpo_name": "name",
                    }
                ),
            ]
        ).sort_values("id:ID").drop_duplicates().to_csv(
            self.nodes_file, index=False, sep="\t"
        )

        # Dump edges
        edges = df[
            [
                "disease_curie",
                "phenotype_curie",
                "evidence",
                "frequency",
                "onset",
                "qualifier",
                "sex",
                "aspect",
                "modifier",
            ]
        ]
        edges = edges.rename(
            columns={
                "disease_curie": ":START_ID",
                "phenotype_curie": ":END_ID",
            }
        )
        edges[":TYPE"] = "has_phenotype"
        edges = edges.sort_values([":START_ID", ":END_ID"])
        edges.drop_duplicates().to_csv(self.edges_file, sep="\t", index=False)


if __name__ == "__main__":
    exporter = HpoExporter()
    exporter.export()
