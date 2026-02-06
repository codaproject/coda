"""
Generate knowledge graph (TSV files) for retrieval terms.

Similar to src/coda/kg/sources pattern, exports nodes.tsv and edges.tsv
for Neo4j-compatible format. Also handles embedding generation internally.
"""
import json
import logging
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from . import SETUP_BASE
from ..retrieval_term import RetrievalTerm

logger = logging.getLogger(__name__)

# Create pystow module for retrieval KG (subfolder of setup)
RETRIEVAL_KG_BASE = SETUP_BASE.module("retrieval_kg")


def _get_embedding_text(term: RetrievalTerm) -> str:
    """
    Get combined text for embedding generation.
    
    Combines name, definition, and synonyms into a single text string.
    
    Parameters
    ----------
    term : RetrievalTerm
        Term to get embedding text for.
    
    Returns
    -------
    str
        Combined text for embedding.
    """
    parts = [term.text]  # Primary name
    if term.definition:
        parts.append(term.definition)
    if term.synonyms:
        parts.extend(term.synonyms)
    return " ".join(parts)


def generate_embeddings(
    terms: List[RetrievalTerm],
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32,
    show_progress: bool = True
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Generate embeddings for a list of retrieval terms.
    
    Parameters
    ----------
    terms : List[RetrievalTerm]
        List of terms to generate embeddings for.
    model_name : str
        SentenceTransformer model name. Defaults to 'all-MiniLM-L6-v2'.
    batch_size : int
        Batch size for embedding generation. Defaults to 32.
    show_progress : bool
        Whether to show progress bar. Defaults to True.
    
    Returns
    -------
    Tuple[np.ndarray, Dict[str, int]]
        Tuple of (embeddings array, term_id_to_index mapping).
        embeddings: shape (n_terms, embedding_dim)
        term_id_to_index: maps term.id to row index in embeddings
    """
    if not terms:
        raise ValueError("terms list cannot be empty")
    
    logger.info(f"Generating embeddings for {len(terms)} terms using model: {model_name}")
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Prepare text for embedding (combine name + definition + synonyms)
    texts = [_get_embedding_text(term) for term in terms]
    
    # Generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    
    # Create mapping from term identifier to embedding index
    term_id_to_index = {
        term.id: idx
        for idx, term in enumerate(terms)
    }
    
    logger.info(f"Generated embeddings: shape {embeddings.shape}")
    
    return embeddings, term_id_to_index


def export_kg(
    terms: List[RetrievalTerm],
    embeddings: np.ndarray,
    term_id_to_index: Dict[str, int],
    output_dir: Optional[Path] = None
) -> Tuple[Path, Path]:
    """
    Export retrieval terms to Neo4j-compatible TSV files (nodes.tsv and edges.tsv).
    
    Similar to ICD10Exporter pattern from src/coda/kg/sources.
    
    Parameters
    ----------
    terms : List[RetrievalTerm]
        List of retrieval terms to export.
    embeddings : np.ndarray
        Embeddings array for the terms.
    term_id_to_index : Dict[str, int]
        Mapping from term identifier to embedding index.
    output_dir : Path, optional
        Directory to save TSV files to. If None, uses pystow default location.
    
    Returns
    -------
    Tuple[Path, Path]
        Tuple of (nodes_file_path, edges_file_path).
    """
    if output_dir is None:
        output_dir = Path(RETRIEVAL_KG_BASE.base)
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    nodes_file = output_dir / 'retrieval_terms_nodes.tsv'
    edges_file = output_dir / 'retrieval_terms_edges.tsv'
    
    logger.info(f"Exporting KG to {output_dir}")
    
    # Build nodes list
    nodes = []
    for idx, term in enumerate(terms):
        # Get embedding for this term
        embedding_idx = term_id_to_index.get(term.id)
        embedding_str = (
            ";".join(embeddings[embedding_idx].astype(str).tolist())
            if embedding_idx is not None
            else ""
        )
        
        # Format synonyms as semicolon-separated string
        synonyms_str = ";".join(term.synonyms) if term.synonyms else ""
        
        nodes.append({
            'id:ID': f"{term.db}:{term.id}",  # CURIE format using term's db and id
            ':LABEL': term.db,
            'identifier': term.id,
            'name': term.text,
            'entry_name': term.entry_name,
            'synonyms': synonyms_str,
            'definition': term.definition or "",
            'db': term.db,
            'status': term.status,
            'source': term.source,
            'embedding:float[]': embedding_str,
        })
    
    # Create nodes DataFrame and save
    nodes_df = pd.DataFrame(nodes)
    nodes_df.sort_values('id:ID').to_csv(nodes_file, sep='\t', index=False)
    logger.info(f"Exported {len(nodes)} nodes to {nodes_file}")
    
    # Create empty edges DataFrame (no relationships for standalone vector store)
    edges_df = pd.DataFrame(columns=[':START_ID', ':END_ID', ':TYPE'])
    edges_df.to_csv(edges_file, sep='\t', index=False)
    logger.info(f"Exported empty edges file to {edges_file}")
    
    return nodes_file, edges_file


def load_kg(
    kg_dir: Optional[Path] = None
) -> Tuple[List[RetrievalTerm], np.ndarray, Dict[str, int]]:
    """
    Load retrieval terms and embeddings from TSV files.
    
    Parameters
    ----------
    kg_dir : Path, optional
        Directory containing TSV files. If None, uses pystow default location.
    
    Returns
    -------
    Tuple[List[RetrievalTerm], np.ndarray, Dict[str, int]]
        Tuple of (terms, embeddings, term_id_to_index).
    
    Raises
    ------
    FileNotFoundError
        If TSV files are not found.
    """
    if kg_dir is None:
        kg_dir = Path(RETRIEVAL_KG_BASE.base)
    else:
        kg_dir = Path(kg_dir)
    
    nodes_file = kg_dir / 'retrieval_terms_nodes.tsv'
    if not nodes_file.exists():
        raise FileNotFoundError(
            f"Nodes file not found: {nodes_file}. "
            f"Run setup to generate KG first."
        )
    
    logger.info(f"Loading KG from {kg_dir}")
    
    # Load nodes TSV
    nodes_df = pd.read_csv(nodes_file, sep='\t')
    logger.info(f"Loaded {len(nodes_df)} nodes from {nodes_file}")
    
    # Reconstruct RetrievalTerm objects and extract embeddings
    terms = []
    embeddings_list = []
    term_id_to_index = {}
    
    for idx, row in nodes_df.iterrows():
        # Parse embedding string back to numpy array
        embedding_str = row.get('embedding:float[]', '')
        if embedding_str and pd.notna(embedding_str):
            embedding = np.array([float(x) for x in embedding_str.split(';')])
        else:
            embedding = None
        
        # Parse synonyms
        synonyms_str = row.get('synonyms', '')
        synonyms = synonyms_str.split(';') if synonyms_str and pd.notna(synonyms_str) else []
        synonyms = [s for s in synonyms if s]  # Remove empty strings
        
        # Create RetrievalTerm
        term = RetrievalTerm(
            norm_text=row.get('name', '').lower().strip(),
            text=row.get('name', ''),
            db=row.get('db', 'RETRIEVAL'),
            id=row.get('identifier', ''),
            entry_name=row.get('entry_name', row.get('name', '')),
            status=row.get('status', 'name'),
            source=row.get('source', 'RAG_GROUNDER'),
            synonyms=synonyms,
            definition=row.get('definition', '') or None,
        )
        terms.append(term)
        
        # Store embedding if available
        if embedding is not None:
            embeddings_list.append(embedding)
            term_id_to_index[term.id] = len(embeddings_list) - 1
    
    # Convert embeddings list to numpy array
    if embeddings_list:
        embeddings = np.array(embeddings_list)
    else:
        raise ValueError("No embeddings found in nodes file")
    
    logger.info(f"Loaded {len(terms)} terms with embeddings: shape {embeddings.shape}")
    
    return terms, embeddings, term_id_to_index


def get_term_index(
    term_id_to_index: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    Get index mapping for terms.
    
    Parameters
    ----------
    term_id_to_index : Dict[str, int], optional
        Mapping from term ID to embedding index. If None, loads from disk.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with 'term_id_to_idx' and 'idx_to_term_id' mappings.
    """
    if term_id_to_index is None:
        _, _, term_id_to_index = load_kg()
    
    return {
        'term_id_to_idx': term_id_to_index,
        'idx_to_term_id': {idx: term_id for term_id, idx in term_id_to_index.items()},
    }
