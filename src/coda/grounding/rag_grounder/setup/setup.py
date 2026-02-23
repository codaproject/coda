import logging
from typing import List

from ..retrieval_term import RetrievalTerm, TermStore
from .generate_index import generate_index

logger = logging.getLogger(__name__)


def setup_retrieval_grounder(
    terms: List[RetrievalTerm],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> TermStore:
    if not terms:
        raise ValueError("terms must not be empty")

    logger.info(f"Starting retrieval grounder setup for {len(terms)} terms...")
    term_store = generate_index(
        terms=terms,
        model_name=model_name,
        batch_size=batch_size,
    )
    logger.info(f"Setup complete. {len(terms)} terms with embeddings in memory")
    return term_store
