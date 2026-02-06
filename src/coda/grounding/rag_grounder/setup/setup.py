"""
Main setup orchestrator for RAG grounder.

Coordinates embedding generation and KG export.
"""
from typing import List, Optional
from pathlib import Path
import logging

from ..retrieval_term import RetrievalTerm
from .generate_kg import generate_embeddings, export_kg

logger = logging.getLogger(__name__)


def setup_retrieval_grounder(
    terms: List[RetrievalTerm],
    output_dir: Optional[Path] = None,
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32
) -> None:
    """
    Set up retrieval grounder by generating embeddings and exporting KG.
    
    Parameters
    ----------
    terms : List[RetrievalTerm]
        List of retrieval terms to set up.
    output_dir : Path, optional
        Directory to save KG files. If None, uses pystow default location.
    model_name : str
        SentenceTransformer model name. Defaults to 'all-MiniLM-L6-v2'.
    batch_size : int
        Batch size for embedding generation. Defaults to 32.
    
    Notes
    -----
    This is a one-time setup step. After running this, use the runtime
    module to load KG and perform retrieval.
    """
    logger.info(f"Starting retrieval grounder setup for {len(terms)} terms...")
    
    # Step 1: Generate embeddings
    embeddings, term_id_to_index = generate_embeddings(
        terms=terms,
        model_name=model_name,
        batch_size=batch_size
    )
    
    # Step 2: Export KG files
    export_kg(
        terms=terms,
        embeddings=embeddings,
        term_id_to_index=term_id_to_index,
        output_dir=output_dir
    )
    
    logger.info("Retrieval grounder setup completed.")
