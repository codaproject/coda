import logging
from pathlib import Path
from typing import List

from ..retrieval_term import RetrievalTerm, TermStore
from .generate_index import generate_index, load_index, save_index

logger = logging.getLogger(__name__)


def setup_retrieval_grounder(
    terms: List[RetrievalTerm],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    cache_path: str | Path | None = None,
    force_rebuild: bool = False,
) -> TermStore:
    if not terms:
        raise ValueError("terms must not be empty")

    logger.info(f"Starting retrieval grounder setup for {len(terms)} terms...")

    resolved_cache_path = Path(cache_path) if cache_path is not None else None

    if resolved_cache_path is not None and not force_rebuild:
        try:
            cached = load_index(resolved_cache_path)
            cached_ids = [term.id for term in cached.terms]
            expected_ids = [term.id for term in terms]
            if cached_ids == expected_ids:
                logger.info("Using cached retrieval index from %s", resolved_cache_path)
                return cached
            logger.warning("Cache term IDs do not match current terms. Regenerating index.")
        except FileNotFoundError:
            logger.info("No cache found at %s. Generating new index.", resolved_cache_path)
        except Exception as exc:
            logger.warning(
                "Failed to load retrieval index cache from %s: %s. Regenerating.",
                resolved_cache_path,
                exc,
            )

    term_store = generate_index(
        terms=terms,
        model_name=model_name,
        batch_size=batch_size,
    )

    if resolved_cache_path is not None:
        try:
            save_index(term_store, resolved_cache_path)
        except Exception as exc:
            logger.warning(
                "Failed to save retrieval index cache to %s: %s",
                resolved_cache_path,
                exc,
            )

    logger.info(f"Setup complete. {len(terms)} terms with embeddings in memory")
    return term_store
