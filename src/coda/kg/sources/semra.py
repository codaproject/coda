import logging

import pandas as pd

from coda import CODA_BASE
from coda.kg.sources import KGSourceExporter, KG_BASE

logger = logging.getLogger(__name__)

SEMRA_PROCESSED_SSSOM_URL = "https://zenodo.org/records/15826693/files/processed.sssom.tsv.gz"

PREFIX_PRIORITY = [
    # ICD codes going from latest to oldest / specialized instances
    'icd11', 'icd10', 'icd10cm', 'icd9', 'icd9cm', 'icdo',
    # Various communicy-curated disease ontologies from general to specialized
    'mondo', 'doid', 'efo', 'omim', 'omim.ps', 'orphanet', 'gard',
    # General-purpose terminologies that include diseases
    'mesh', 'ncit', 'umls'
]



def _discover_kg_prefixes() -> set[str]:
    """Scan existing *_nodes.tsv files in kg/ and collect all CURIE prefixes."""
    prefixes = set()
    for nodes_file in KG_BASE.glob("*_nodes.tsv.gz"):
        if nodes_file.name.startswith("semra"):
            continue
        try:
            ids = pd.read_csv(nodes_file, sep="\t", usecols=["id:ID"],
                              low_memory=False)["id:ID"].dropna()
            prefixes.update(ids.str.split(":").str[0].unique())
        except Exception as e:
            logger.warning("Could not read %s: %s", nodes_file.name, e)
    logger.info("Discovered KG prefixes: %s", sorted(prefixes))
    return prefixes


class SemraExporter(KGSourceExporter):
    name = "semra"

    def export(self):
        kg_prefixes = _discover_kg_prefixes()
        if not kg_prefixes:
            logger.warning("No KG node files found in %s, nothing to map.", KG_BASE)
            return

        sssom_path = CODA_BASE.ensure("kg", name="processed.sssom.tsv.gz",
                                      url=SEMRA_PROCESSED_SSSOM_URL)
        logger.info("Loading SEMRA SSSOM from %s ...", sssom_path)
        df = pd.read_csv(sssom_path, sep="\t", compression="gzip",
                         low_memory=False)
        df = df.dropna(subset=["subject_id", "object_id"])

        # Keep only edges where both ends match a KG prefix
        df["subject_prefix"] = df["subject_id"].str.split(":").str[0]
        df["object_prefix"] = df["object_id"].str.split(":").str[0]
        df = df[df["subject_prefix"].isin(kg_prefixes) & df["object_prefix"].isin(kg_prefixes)]

        # Drop self-mappings and duplicates
        df = df[df["subject_id"] != df["object_id"]]
        df = df.drop_duplicates(subset=["subject_id", "object_id"])

        # Deduplicate symmetric pairs, keep one edge per pair
        df["_lo"] = df[["subject_id", "object_id"]].min(axis=1)
        df["_hi"] = df[["subject_id", "object_id"]].max(axis=1)
        # We need to ensure that we keep mappings where the subject
        # has lower priority than the object, so we sort by prefix priority and
        # then by subject/object ID
        df["subject_prefix_priority"] = df["subject_prefix"].apply(
            lambda p: len(PREFIX_PRIORITY) - PREFIX_PRIORITY.index(p) \
                          if p in PREFIX_PRIORITY else 0
        )
        df["object_prefix_priority"] = df["object_prefix"].apply(
            lambda p: len(PREFIX_PRIORITY) - PREFIX_PRIORITY.index(p) \
                if p in PREFIX_PRIORITY else 0
        )
        df = df.sort_values(
            ["_lo", "_hi", "subject_prefix_priority", "object_prefix_priority",
             "subject_id", "object_id"]
        )
        df = df.drop_duplicates(subset=["_lo", "_hi"]).drop(columns=["_lo", "_hi"])

        logger.info("Filtered to %d cross-ontology mappings", len(df))

        # Build edges
        predicate = df["predicate_id"].fillna("skos:relatedMatch") \
            if "predicate_id" in df.columns else "skos:relatedMatch"
        edges = pd.DataFrame({
            ":START_ID": df["subject_id"].values,
            ":END_ID": df["object_id"].values,
            ":TYPE": predicate,
        })
        edges.sort_values([":START_ID", ":END_ID"]).to_csv(self.edges_file, sep="\t", index=False)
        logger.info("Wrote %d edges to %s", len(edges), self.edges_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exporter = SemraExporter()
    exporter.export()