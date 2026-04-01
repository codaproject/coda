"""
Generate embeddings for retrieval terms.

Uses SentenceTransformer. Returns terms and embeddings in memory.
"""
import logging
import pickle
from pathlib import Path
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


def save_index(term_store: TermStore, path: str | Path) -> None:
    """
    Save a term index to disk.

    Parameters
    ----------
    term_store : TermStore
        Terms and embeddings to persist.
    path : str | Path
        Target file path.
    """
    if not term_store.terms:
        raise ValueError("term_store.terms must not be empty")

    embeddings = np.asarray(term_store.embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(
            f"term_store.embeddings must be 2D, got shape {embeddings.shape}"
        )
    if embeddings.shape[0] != len(term_store.terms):
        raise ValueError(
            "term_store.terms and term_store.embeddings row count must match"
        )

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "terms": term_store.terms,
        "embeddings": embeddings,
    }
    with target_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(
        "Saved index cache to %s (%d terms, shape=%s)",
        target_path,
        len(term_store.terms),
        embeddings.shape,
    )


def load_index(path: str | Path) -> TermStore:
    """
    Load a term index from disk.

    Parameters
    ----------
    path : str | Path
        File path containing serialized terms and embeddings.

    Returns
    -------
    TermStore
        Loaded in-memory term store.
    """
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"index cache not found: {source_path}")

    with source_path.open("rb") as fh:
        payload = pickle.load(fh)

    terms = payload.get("terms")
    embeddings = np.asarray(payload.get("embeddings"), dtype=np.float32)

    if not isinstance(terms, list) or not all(
        isinstance(term, RetrievalTerm) for term in terms
    ):
        raise ValueError("cached index contains invalid terms")
    if embeddings.ndim != 2:
        raise ValueError(
            f"cached embeddings must be 2D, got shape {embeddings.shape}"
        )
    if embeddings.shape[0] != len(terms):
        raise ValueError("cached terms and embeddings row count do not match")

    logger.info(
        "Loaded index cache from %s (%d terms, shape=%s)",
        source_path,
        len(terms),
        embeddings.shape,
    )
    return TermStore(terms=terms, embeddings=embeddings)
