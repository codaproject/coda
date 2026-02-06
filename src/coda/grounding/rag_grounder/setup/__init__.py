"""
Setup module for RAG grounder.

Handles one-time setup tasks:
- Generating embeddings for retrieval terms
- Saving embeddings and metadata to disk
"""
from .. import RAG_GROUNDER_BASE

# Create pystow module for setup data
SETUP_BASE = RAG_GROUNDER_BASE.module("setup")

# Import after SETUP_BASE is defined to avoid circular imports
from .generate_kg import (
    generate_embeddings,
    export_kg,
    load_kg,
    get_term_index,
    RETRIEVAL_KG_BASE,
)
from .setup import setup_retrieval_grounder

__all__ = [
    'SETUP_BASE',
    'generate_embeddings',
    'export_kg',
    'load_kg',
    'get_term_index',
    'RETRIEVAL_KG_BASE',
    'setup_retrieval_grounder',
]
