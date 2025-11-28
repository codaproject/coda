"""
ICD-10 code retrieval using semantic embeddings.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from openacme.generate_embeddings.generate_embeddings import load_embeddings, get_code_index


class ICD10Retriever:
    """
    Efficient ICD-10 code retriever using semantic embeddings.

    Loads embeddings and model once for reuse across multiple queries.
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        """Initialize ICD-10 retriever.

        Parameters
        ----------
        model_name : str
            SentenceTransformer model name. Defaults to 'all-MiniLM-L6-v2'.
        """
        embeddings, definitions_data = load_embeddings()
        self.embeddings = embeddings
        self.definitions_data = definitions_data
        # Generate code index using openacme's helper function
        self.code_index = get_code_index(definitions_data)
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the SentenceTransformer model."""
        if self._model is None:
            print(f"Loading SentenceTransformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def retrieve(
        self,
        clinical_text: str,
        top_k: int = 10,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k most similar ICD-10 codes for clinical text.

        Parameters
        ----------
        clinical_text : str
            Clinical description or evidence text.
        top_k : int
            Number of top codes to return. Defaults to 10.
        min_similarity : float
            Minimum similarity threshold (0.0 to 1.0). Defaults to 0.0.

        Returns
        -------
        list of dict
            List of dictionaries with code, similarity, name, and definition.
        """
        if not clinical_text or not clinical_text.strip():
            return []

        # Generate embedding for clinical text
        clinical_embedding = self.model.encode(
            [clinical_text],
            normalize_embeddings=True
        )

        # Calculate cosine similarity
        similarities = cosine_similarity(clinical_embedding, self.embeddings)[0]

        # Filter by minimum similarity
        valid_indices = np.where(similarities >= min_similarity)[0]

        if len(valid_indices) == 0:
            return []

        # Get top-k most similar codes
        top_indices = similarities[valid_indices].argsort()[-top_k:][::-1]
        top_indices = valid_indices[top_indices]

        results = []
        for idx in top_indices:
            # get_code_index returns idx_to_code as a list, not a dict
            code = self.code_index['idx_to_code'][idx]
            similarity = float(similarities[idx])
            name = self.definitions_data.get(code, {}).get('name', f'Code: {code}')
            definition = self.definitions_data.get(code, {}).get('definition', '')

            results.append({
                'code': code,
                'similarity': similarity,
                'name': name,
                'definition': definition
            })

        return results

    def get_code_name(self, code: str) -> str:
        """Get human-readable name for an ICD-10 code.

        Parameters
        ----------
        code : str
            ICD-10 code.

        Returns
        -------
        str
            Code name or error message if code not found.
        """
        if code not in self.definitions_data:
            return f"Unknown code: {code}"
        return self.definitions_data[code].get('name', f'Code: {code}')

    def get_code_definition(self, code: str) -> str:
        """Get definition for an ICD-10 code.

        Parameters
        ----------
        code : str
            ICD-10 code.

        Returns
        -------
        str
            Code definition or empty string if code not found.
        """
        if code not in self.definitions_data:
            return ""
        return self.definitions_data[code].get('definition', '')

