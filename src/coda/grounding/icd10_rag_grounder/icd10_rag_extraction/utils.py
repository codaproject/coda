"""
Utility functions for validation and data processing.
"""

import re
from typing import Optional, Dict, Any
from pathlib import Path
import json

from openacme.icd10.generate_embeddings import EMBEDDINGS_BASE


# Cache for definitions data to avoid reloading
_cached_definitions_data: Optional[Dict[str, Any]] = None


def validate_icd10_code(
    code: str,
    check_existence: bool = False,
    definitions_data: Optional[Dict[str, Any]] = None
) -> bool:
    """Validate ICD-10 code format and optionally check existence in embedding space."""
    if not code or not isinstance(code, str):
        return False

    code = code.strip().upper()

    # Pattern: Letter followed by 2 digits, optionally followed by . and more digits
    pattern = r"^[A-Z][0-9]{2}(\.[0-9]+)?$"
    if not re.match(pattern, code):
        return False

    if check_existence:
        global _cached_definitions_data
        if definitions_data is None:
            if _cached_definitions_data is None:
                _cached_definitions_data = load_icd10_definitions()
            definitions_data = _cached_definitions_data
        return code in definitions_data

    return True


def load_icd10_definitions(definitions_file: Optional[Path] = None) -> Dict[str, Any]:
    """Load ICD-10 code definitions from JSON file.

    Parameters
    ----------
    definitions_file : pathlib.Path, optional
        Path to definitions JSON file. Defaults to openacme's icd10_embeddings
        directory.

    Returns
    -------
    dict
        Dictionary mapping codes to definition data.
    """
    if definitions_file is None:
        # Use openacme's EMBEDDINGS_BASE to get the path
        definitions_file = Path(EMBEDDINGS_BASE.base) / 'icd10_code_to_definition.json'

    definitions_file = Path(definitions_file)
    if not definitions_file.exists():
        raise FileNotFoundError(f"Definitions file not found: {definitions_file}")

    with open(definitions_file, 'r', encoding='utf-8') as f:
        return json.load(f)
