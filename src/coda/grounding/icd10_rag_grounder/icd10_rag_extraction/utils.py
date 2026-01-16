"""
Utility functions for validation and data processing (mention-based pipeline).
"""

import re
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

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
    """Load ICD-10 code definitions from JSON file."""
    if definitions_file is None:
        definitions_file = Path(EMBEDDINGS_BASE.base) / "icd10_code_to_definition.json"

    definitions_file = Path(definitions_file)
    if not definitions_file.exists():
        raise FileNotFoundError(f"Definitions file not found: {definitions_file}")

    with open(definitions_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_icd10_name(code: str, definitions_data: Optional[Dict[str, Any]] = None) -> str:
    """Get human-readable name for an ICD-10 code."""
    if not code or not isinstance(code, str):
        return "Unknown code: "

    code = code.strip().upper()

    if definitions_data is None:
        definitions_data = load_icd10_definitions()

    if code not in definitions_data:
        return f"Unknown code: {code}"

    return definitions_data[code].get("name", f"Code: {code}")


def validate_mention_extraction_result(result: Dict[str, Any]) -> bool:
    """Validate structure of mention extraction result."""
    if not isinstance(result, dict):
        return False
    if "Mentions" not in result:
        return False
    if not isinstance(result["Mentions"], list):
        return False
    return True


def validate_disease_extraction_result(result: Dict[str, Any]) -> bool:
    """Validate structure of disease extraction result."""
    if not isinstance(result, dict):
        return False
    if "Diseases" not in result:
        return False
    if not isinstance(result["Diseases"], list):
        return False
    return True


def mention_to_retrieval_text(mention: str) -> str:
    """Normalize a mention span into the text used for retrieval."""
    if not mention or not isinstance(mention, str):
        return ""
    return mention.strip()
