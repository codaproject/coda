"""
Term retrieval using neo4j vector search.
"""
import logging
from typing import List, Tuple

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from coda.runtime_config import get_kg_url

from .types import RetrievalTerm

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves ontology terms from neo4j using vector similarity search.
    """

    def __init__(
        self,
        ontology: str,
        model_name: str,
        top_k: int,
        min_similarity: float
    ):
        self.driver = GraphDatabase.driver(get_kg_url(), auth=None)
        self.ontology = ontology
        self.model_name = model_name
        self.top_k = top_k
        self.min_similarity = min_similarity
        # The vector index is created at KG startup (see coda.kg.vector_index);
        # here we only load the model used to encode queries.
        self._model = SentenceTransformer(self.model_name)

    def retrieve(
        self,
        query_text: str
    ) -> List[Tuple[RetrievalTerm, float]]:
        if not query_text or not query_text.strip():
            return []

        query_embedding = self._model.encode(
            query_text,
            normalize_embeddings=True,
        ).tolist()

        index_name = f"{self.ontology}_embedding"

        with self.driver.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
                YIELD node, score
                WHERE score >= $min_similarity
                RETURN node.id AS id, node.name AS name, score
                """,
                index_name=index_name,
                top_k=self.top_k,
                query_embedding=query_embedding,
                min_similarity=self.min_similarity,
            )
            return [
                (RetrievalTerm(id=record["id"], name=record["name"]), record["score"])
                for record in result
            ]
