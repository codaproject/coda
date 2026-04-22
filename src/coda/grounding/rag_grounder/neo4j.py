import os

from neo4j import GraphDatabase

from .retrieval_term import RetrievalTerm

_DEFAULT_NEO4J_URL = "bolt://localhost:7687"


def load_retrieval_terms(ontology: str, neo4j_url: str | None = None) -> list[RetrievalTerm]:
    url = neo4j_url or os.environ.get("NEO4J_URL", _DEFAULT_NEO4J_URL)
    driver = GraphDatabase.driver(url, auth=None)

    query = f"MATCH (n:{ontology}) RETURN n.ID AS id, n.name AS name"

    terms: list[RetrievalTerm] = []
    with driver.session() as session:
        for record in session.run(query):
            node_id = record["id"]
            name = record["name"]
            if not node_id or not name:
                continue
            terms.append(
                RetrievalTerm(
                    id=node_id,
                    name=name,
                    definition=None,
                    synonyms=None,
                )
            )

    driver.close()
    terms.sort(key=lambda t: t.id)
    return terms
