"""
Generate embeddings for retrieval terms.

Uses SentenceTransformer. Returns terms and embeddings in memory.
"""
import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from ..retrieval_term import RetrievalTerm, TermStore

logger = logging.getLogger(__name__)


def _text_for_embedding(term: RetrievalTerm) -> str:
    """Build text to embed: name + definition for richer representation."""
    parts = [term.name]
    if term.definition:
        parts.append(term.definition)
    return " ".join(parts)


def generate_embeddings(
    terms: List[RetrievalTerm],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Generate embeddings for retrieval terms.

    Parameters
    ----------
    terms : List[RetrievalTerm]
        Terms to embed.
    model_name : str
        SentenceTransformer model name.
    batch_size : int
        Batch size for encoding.
    show_progress : bool
        Whether to show tqdm progress.

    Returns
    -------
    np.ndarray
        Embedding matrix, shape (n_terms, dim).
    """
    if not terms:
        raise ValueError("terms must not be empty")

    logger.info(f"Generating embeddings for {len(terms)} terms using model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [_text_for_embedding(t) for t in terms]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    logger.info(f"Generated embeddings: shape {embeddings.shape}")
    return embeddings


def generate_index(
    terms: List[RetrievalTerm],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> TermStore:
    """
    Generate embeddings for retrieval terms.

    Parameters
    ----------
    terms : List[RetrievalTerm]
        Terms to embed.
    model_name : str
        SentenceTransformer model name.
    batch_size : int
        Batch size for encoding.

    Returns
    -------
    TermStore
        Terms and embeddings in memory.
    """
    embeddings = generate_embeddings(
        terms=terms,
        model_name=model_name,
        batch_size=batch_size,
    )
    return TermStore(terms=terms, embeddings=embeddings)
