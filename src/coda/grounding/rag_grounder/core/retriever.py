"""
Term retrieval using semantic embeddings.

Uses terms and embeddings from in-memory term store.
"""
import logging
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..retrieval_term import RetrievalTerm, TermStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Efficient term retriever using semantic embeddings.

    Uses terms and embeddings from in-memory term store.
    """

    def __init__(
        self,
        term_store: TermStore,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """Initialize retriever.

        Parameters
        ----------
        term_store : TermStore
            Terms and embeddings from setup.
        model_name : str
            SentenceTransformer model name. Defaults to 'all-MiniLM-L6-v2'.
            Should match the model used during setup.
        """
        self.term_store = term_store
        self.model_name = model_name
        self._model = None

    def retrieve(
        self,
        query_text: str,
        top_k: int = 10,
        min_similarity: float = 0.5,
    ) -> List[Tuple[RetrievalTerm, float]]:
        """Retrieve top-k most similar terms for query text.

        Parameters
        ----------
        query_text : str
            Query text to search for.
        top_k : int
            Number of top terms to return. Defaults to 10.
        min_similarity : float
            Minimum similarity threshold (0.0 to 1.0). Defaults to 0.0.

        Returns
        -------
        list of tuple (RetrievalTerm, float)
            List of tuples containing (RetrievalTerm, similarity_score), ordered by similarity descending.
        """
        if not query_text or not query_text.strip():
            return []

        terms = self.term_store.terms
        embeddings = self.term_store.embeddings

        if self._model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)

        if len(embeddings) == 0:
            return []

        # Generate embedding for query text
        query_embedding = self._model.encode(
            [query_text],
            normalize_embeddings=True,
        )

        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, embeddings)[0]

        # Filter by minimum similarity
        valid_indices = np.where(similarities >= min_similarity)[0]

        if len(valid_indices) == 0:
            return []

        # Get top-k most similar terms
        top_indices = similarities[valid_indices].argsort()[-top_k:][::-1]
        top_indices = valid_indices[top_indices]

        results = []
        for idx in top_indices:
            similarity_score = float(similarities[idx])
            results.append((terms[idx], similarity_score))

        return results
