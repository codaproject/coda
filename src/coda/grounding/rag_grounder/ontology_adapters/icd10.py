"""
ICD-10 ontology adapter for RAG retrieval terms.
Fetches terms from the Neo4j knowledge graph at runtime.
"""

from __future__ import annotations

import os

from neo4j import GraphDatabase

from ..retrieval_term import RetrievalTerm

_DEFAULT_NEO4J_URL = "bolt://localhost:7687"


def load_icd10_retrieval_terms(
    prefix: str = "icd10",
    neo4j_url: str | None = None,
) -> list[RetrievalTerm]:
    url = neo4j_url or os.environ.get("NEO4J_URL", _DEFAULT_NEO4J_URL)
    driver = GraphDatabase.driver(url, auth=None)

    query = "MATCH (n:icd10) RETURN n.code AS code, n.name AS name, n.rubrics AS rubrics"

    terms: list[RetrievalTerm] = []
    with driver.session() as session:
        for record in session.run(query):
            node_id = f"{prefix}:{record['code']}"
            name = record["name"] or node_id
            rubrics = record["rubrics"] or {}

            synonyms: list[str] = []
            if isinstance(rubrics, dict):
                for key, values in rubrics.items():
                    if key == "preferred":
                        continue
                    if isinstance(values, list):
                        synonyms.extend(v for v in values if v)
                    elif values:
                        synonyms.append(str(values))
            synonyms = list(dict.fromkeys(synonyms))

            terms.append(
                RetrievalTerm(
                    id=node_id,
                    name=name,
                    definition=synonyms[0] if synonyms else None,
                    synonyms=synonyms or None,
                )
            )

    driver.close()
    terms.sort(key=lambda t: t.id)
    return terms
