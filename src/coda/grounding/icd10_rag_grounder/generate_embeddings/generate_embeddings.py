"""Generate embeddings for ICD-10 codes from definitions."""

import json
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from coda.resources import get_resource_path
from .config import DEFAULT_MODEL, DEFAULT_BATCH_SIZE

# Default output directory in resources
_DEFAULT_OUTPUT_DIR = Path(get_resource_path('icd10_embeddings'))


def load_icd10_definitions(json_file, verbose=True):
    """Load ICD-10 code to definition mappings."""
    if verbose:
        print(f"Loading ICD-10 definitions from {json_file}...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to list of (code, definition) tuples
    codes = []
    definitions = []
    metadata = []
    
    for code, entry in sorted(data.items()):
        codes.append(code)
        definitions.append(entry['definition'])
        metadata.append({
            'code': code,
            'name': entry['name'],
            'source': entry['source'],
            'has_definition': entry['has_definition']
        })
    
    if verbose:
        print(f"  ✓ Loaded {len(codes)} ICD-10 codes")
    
    return codes, definitions, metadata


def generate_embeddings(
    definitions,
    model_name=DEFAULT_MODEL,
    batch_size=DEFAULT_BATCH_SIZE,
    normalize=True,
    verbose=True
):
    """
    Generate embeddings for definitions using sentence transformers.
    
    Args:
        definitions: List of definition strings
        model_name: Sentence transformer model name
        batch_size: Batch size for encoding
        normalize: Whether to normalize embeddings to unit vectors
        verbose: Print progress messages
    
    Returns:
        numpy array of embeddings
    """
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers"
        )
    
    if verbose:
        print(f"\nLoading sentence transformer model: {model_name}...")
    
    model = SentenceTransformer(model_name)
    
    if verbose:
        print(f"  ✓ Model loaded (max_seq_length: {model.max_seq_length})")
        print(f"\nGenerating embeddings (batch_size={batch_size})...")
    
    embeddings = model.encode(
        definitions,
        batch_size=batch_size,
        show_progress_bar=verbose,
        convert_to_numpy=True,
        normalize_embeddings=normalize
    )
    
    if verbose:
        print(f"  ✓ Generated embeddings shape: {embeddings.shape}")
        print(f"  ✓ Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings


def save_embeddings(codes, embeddings, metadata, output_dir, definitions_json=None, definitions_csv=None, verbose=True):
    """
    Save embeddings and metadata.
    
    Args:
        codes: List of ICD-10 codes
        embeddings: numpy array of embeddings
        metadata: List of metadata dictionaries
        output_dir: Output directory path
        definitions_json: Path to source definitions JSON file (to copy)
        definitions_csv: Path to source definitions CSV file (to copy)
        verbose: Print progress messages
    
    Returns:
        Tuple of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if verbose:
        print(f"\nSaving embeddings to {output_dir}...")
    
    # Save embeddings as numpy array
    embeddings_file = output_dir / 'embeddings.npy'
    np.save(embeddings_file, embeddings)
    if verbose:
        print(f"  ✓ Saved embeddings: {embeddings_file}")
    
    # Save code index mapping
    code_to_idx = {code: idx for idx, code in enumerate(codes)}
    idx_to_code = codes
    
    code_index_file = output_dir / 'code_index.json'
    with open(code_index_file, 'w', encoding='utf-8') as f:
        json.dump({
            'code_to_idx': code_to_idx,
            'idx_to_code': idx_to_code
        }, f, indent=2)
    if verbose:
        print(f"  ✓ Saved code index: {code_index_file}")
    
    # Copy definitions JSON file (if not already in output_dir)
    if definitions_json:
        definitions_json_src = Path(definitions_json)
        definitions_json_dst = output_dir / 'icd10_code_to_definition.json'
        if definitions_json_src.exists() and definitions_json_src != definitions_json_dst:
            import shutil
            shutil.copy2(definitions_json_src, definitions_json_dst)
            if verbose:
                print(f"  ✓ Copied definitions JSON: {definitions_json_dst}")
        elif definitions_json_src == definitions_json_dst:
            if verbose:
                print(f"  ✓ Definitions JSON already in output directory")
    
    # Copy definitions CSV file (if not already in output_dir)
    if definitions_csv:
        definitions_csv_src = Path(definitions_csv)
        definitions_csv_dst = output_dir / 'icd10_code_to_definition.csv'
        if definitions_csv_src.exists() and definitions_csv_src != definitions_csv_dst:
            import shutil
            shutil.copy2(definitions_csv_src, definitions_csv_dst)
            if verbose:
                print(f"  ✓ Copied definitions CSV: {definitions_csv_dst}")
        elif definitions_csv_src == definitions_csv_dst:
            if verbose:
                print(f"  ✓ Definitions CSV already in output directory")
    
    return embeddings_file, code_index_file


def load_embeddings(output_dir):
    """
    Load embeddings and code index from output directory.
    
    Args:
        output_dir: Directory containing embedding files
    
    Returns:
        Tuple of (embeddings, code_index, definitions_data)
    """
    output_dir = Path(output_dir)
    
    # Load embeddings
    embeddings_file = output_dir / 'embeddings.npy'
    embeddings = np.load(embeddings_file)
    
    # Load code index
    code_index_file = output_dir / 'code_index.json'
    with open(code_index_file, 'r') as f:
        code_index = json.load(f)
    
    # Load definitions JSON
    definitions_file = output_dir / 'icd10_code_to_definition.json'
    with open(definitions_file, 'r') as f:
        definitions_data = json.load(f)
    
    return embeddings, code_index, definitions_data


def generate_icd10_embeddings(
    output_dir=None,
    model_name=DEFAULT_MODEL,
    batch_size=DEFAULT_BATCH_SIZE,
    icd10_zip_path=None,
    mrconso_path=None,
    mrdef_path=None,
    verbose=True
):
    """
    Complete pipeline: Map definitions -> Generate embeddings -> Save.
    
    This is a high-level function that:
    1. Calls map_icd10_to_definitions() to create code-to-definition mappings
    2. Generates embeddings from definitions
    3. Saves everything to output_dir
    
    Args:
        output_dir: Output directory for all files (embeddings, definitions, index)
        model_name: Sentence transformer model name
        batch_size: Batch size for encoding
        icd10_zip_path: Path to ICD-10 XML zip file (optional, uses default if None)
        mrconso_path: Path to MRCONSO.RRF file (optional, uses default if None)
        mrdef_path: Path to MRDEF.RRF file (optional, uses default if None)
        verbose: Print progress messages
    
    Returns:
        Tuple of output file paths (embeddings_file, code_index_file)
    """
    from .map_definitions import map_icd10_to_definitions
    
    # Use default output directory if not specified
    if output_dir is None:
        output_dir = _DEFAULT_OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
    
    if verbose:
        print("=" * 70)
        print("ICD-10 Code -> Definition -> Embedding Pipeline")
        print("=" * 70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Step 1: Map codes to definitions (saves to output_dir)
    if verbose:
        print("\nStep 1: Mapping ICD-10 codes to definitions")
        print("-" * 70)
    
    definitions_json = output_dir / 'icd10_code_to_definition.json'
    definitions_csv = output_dir / 'icd10_code_to_definition.csv'
    
    icd10_data = map_icd10_to_definitions(
        icd10_zip_path=icd10_zip_path,
        mrconso_path=mrconso_path,
        mrdef_path=mrdef_path,
        output_json=str(definitions_json),
        output_csv=str(definitions_csv),
        verbose=verbose
    )
    
    # Step 2: Load definitions and generate embeddings
    if verbose:
        print("\nStep 2: Generating embeddings")
        print("-" * 70)
    
    codes, definitions, metadata = load_icd10_definitions(definitions_json, verbose=verbose)
    
    # Generate embeddings
    embeddings = generate_embeddings(
        definitions,
        model_name=model_name,
        batch_size=batch_size,
        verbose=verbose
    )
    
    # Step 3: Save embeddings and code index
    output_files = save_embeddings(
        codes, 
        embeddings, 
        metadata, 
        output_dir,
        definitions_json=str(definitions_json),
        definitions_csv=str(definitions_csv),
        verbose=verbose
    )
    
    if verbose:
        print("\n" + "=" * 70)
        print("✓ Pipeline Complete!")
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  Total codes processed: {len(codes):,}")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"\nAll files saved to: {output_dir}/")
        print(f"  - embeddings.npy")
        print(f"  - code_index.json")
        print(f"  - icd10_code_to_definition.json")
        print(f"  - icd10_code_to_definition.csv")
    
    return output_files

