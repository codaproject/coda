"""
ICD-10 code retrieval using semantic embeddings.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import load_icd10_definitions


class ICD10Retriever:
    """
    Efficient ICD-10 code retriever using semantic embeddings.
    
    Loads embeddings and model once for reuse across multiple queries.
    """
    
    def __init__(
        self,
        embeddings_dir: str = 'data/icd10_embeddings',
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize retriever with embeddings and model.
        
        Args:
            embeddings_dir: Directory containing embeddings.npy and code_index.json
            model_name: SentenceTransformer model name
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.model_name = model_name
        
        # Load embeddings and index
        self._load_embeddings()
        
        # Load definitions
        definitions_file = self.embeddings_dir / 'icd10_code_to_definition.json'
        self.definitions_data = load_icd10_definitions(definitions_file)
        
        # Initialize model (lazy loading)
        self._model = None
    
    def _load_embeddings(self):
        """Load embeddings and code index from disk."""
        embeddings_file = self.embeddings_dir / 'embeddings.npy'
        index_file = self.embeddings_dir / 'code_index.json'
        
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        self.embeddings = np.load(embeddings_file)
        
        with open(index_file, 'r', encoding='utf-8') as f:
            self.code_index = json.load(f)
        
        print(f"Loaded {len(self.embeddings):,} ICD-10 code embeddings")
    
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
        """
        Retrieve top-k most similar ICD-10 codes for clinical text.
        
        Args:
            clinical_text: Clinical description or evidence text
            top_k: Number of top codes to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
        
        Returns:
            List of dictionaries with code, similarity, name, and definition
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
        """
        Get human-readable name for an ICD-10 code.
        
        Args:
            code: ICD-10 code
        
        Returns:
            Code name or error message
        """
        if code not in self.definitions_data:
            return f"Unknown code: {code}"
        return self.definitions_data[code].get('name', f'Code: {code}')
    
    def get_code_definition(self, code: str) -> str:
        """
        Get definition for an ICD-10 code.
        
        Args:
            code: ICD-10 code
        
        Returns:
            Code definition or empty string
        """
        if code not in self.definitions_data:
            return ""
        return self.definitions_data[code].get('definition', '')

