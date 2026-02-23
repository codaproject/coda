"""Setup module: generate embeddings for retrieval terms."""
from .setup import setup_retrieval_grounder
from .generate_index import generate_index, generate_embeddings

__all__ = [
    "setup_retrieval_grounder",
    "generate_index",
    "generate_embeddings",
]
