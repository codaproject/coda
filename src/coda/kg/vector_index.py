"""Create Neo4j vector indexes for embedded KG nodes.

Vector indexes are created via Cypher against a running Neo4j server, so this
cannot be done during the offline `neo4j-admin database import`. It is run once
at KG container startup, after Neo4j is up.

The set of node labels (and the properties holding their embeddings) is declared
explicitly in EMBEDDED_LABELS. For each, a `{label}_{property}` vector index is
created, with the dimension inferred from the already-imported embeddings (no
embedding model is loaded).
"""
import logging
import os

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

SIMILARITY_FUNCTION = "cosine"

# Node labels that carry embeddings, mapped to the properties holding them.
# Keep in sync with the embedding columns written during KG export.
EMBEDDED_LABELS = {
    "icd10": ["embedding"],
    "icd11": ["embedding"],
}


def _embedding_dim(session, label, prop):
    """Read the embedding dimension for a label/property from imported data."""
    record = session.run(
        f"""
        MATCH (n:`{label}`)
        WHERE n.`{prop}` IS NOT NULL
        RETURN size(n.`{prop}`) AS dim
        LIMIT 1
        """
    ).single()
    return record["dim"] if record is not None else None


def ensure_vector_indexes(neo4j_url=None):
    """Ensure a vector index exists for every label/property in EMBEDDED_LABELS."""
    neo4j_url = neo4j_url or os.getenv("CODA_KG_URL", "bolt://localhost:7687")
    driver = GraphDatabase.driver(neo4j_url, auth=None)
    try:
        with driver.session() as session:
            for label, props in EMBEDDED_LABELS.items():
                for prop in props:
                    dim = _embedding_dim(session, label, prop)
                    if dim is None:
                        logger.warning(
                            "No embeddings found on '%s.%s'; skipping index",
                            label, prop,
                        )
                        continue
                    index_name = f"{label}_{prop}"
                    session.run(
                        f"""
                        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                        FOR (n:`{label}`) ON (n.`{prop}`)
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: {dim},
                                `vector.similarity_function`: '{SIMILARITY_FUNCTION}'
                            }}
                        }}
                        """
                    )
                    logger.info("Vector index '%s' ready (dim=%d)", index_name, dim)
            session.run("CALL db.awaitIndexes()")
    finally:
        driver.close()


if __name__ == "__main__":
    ensure_vector_indexes()
