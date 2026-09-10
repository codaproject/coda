"""Pluggable text embedding backends for CODA inference agents."""
from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbedder(ABC):
    """Interface for embedding text into fixed-size vectors.

    dim : int
        Length of the embedding vectors produced by this embedder.
    """

    dim: int

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a single string into a 1D vector of length `dim`."""

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings into a 2D array of shape (len(texts), dim)."""


class SentenceTransformerEmbedder(BaseEmbedder):
    """L2-normalized sentence embeddings via sentence-transformers.

    Lazily loads the underlying model on first use, so importing this module
    does not require sentence-transformers unless the embedder is actually
    instantiated. Empty strings embed to the zero vector rather than being
    passed through the model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        model = self.model
        if hasattr(model, "get_embedding_dimension"):
            return model.get_embedding_dimension()
        return model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        mask = [bool((text or "").strip()) for text in texts]
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        non_empty = [text for text, keep in zip(texts, mask) if keep]
        if non_empty:
            encoded = np.asarray(
                self.model.encode(non_empty, normalize_embeddings=True),
                dtype=np.float32,
            )
            it = iter(encoded)
            for i, keep in enumerate(mask):
                if keep:
                    vectors[i] = next(it)
        return vectors
