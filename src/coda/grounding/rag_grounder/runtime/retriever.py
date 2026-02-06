"""
Term retrieval using semantic embeddings.

Loads embeddings from KG and performs semantic search for retrieval terms.
"""

import logging
import numpy as np
from typing import List, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..setup import load_kg
from ..retrieval_term import RetrievalTerm

logger = logging.getLogger(__name__)


class Retriever:
    """
    Efficient term retriever using semantic embeddings.
    
    Loads embeddings and terms from KG files once for reuse across multiple queries.
    """
    
    def __init__(
        self,
        kg_dir: Optional[str] = None,
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        """Initialize retriever.
        
        Parameters
        ----------
        kg_dir : str, optional
            Directory containing KG TSV files. If None, uses pystow default location.
        model_name : str
            SentenceTransformer model name. Defaults to 'all-MiniLM-L6-v2'.
            Should match the model used during setup.
        """
        self.kg_dir = kg_dir
        self.model_name = model_name
        self._model = None
        self._terms = None
        self._embeddings = None
    
    def retrieve(
        self,
        query_text: str,
        top_k: int = 10,
        min_similarity: float = 0.0
    ) -> List[RetrievalTerm]:
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
        list of RetrievalTerm
            List of RetrievalTerm objects, ordered by similarity descending.
        """
        if not query_text or not query_text.strip():
            return []
        
        # Load KG and model if needed
        if self._terms is None:
            kg_path = Path(self.kg_dir) if self.kg_dir else None
            self._terms, self._embeddings, _ = load_kg(kg_path)
        
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        
        if self._embeddings is None or len(self._embeddings) == 0:
            return []
        
        # Generate embedding for query text
        query_embedding = self._model.encode(
            [query_text],
            normalize_embeddings=True
        )
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self._embeddings)[0]
        
        # Filter by minimum similarity
        valid_indices = np.where(similarities >= min_similarity)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Get top-k most similar terms
        top_indices = similarities[valid_indices].argsort()[-top_k:][::-1]
        top_indices = valid_indices[top_indices]
        
        results = []
        for idx in top_indices:
            results.append(self._terms[idx])
        
        return results
