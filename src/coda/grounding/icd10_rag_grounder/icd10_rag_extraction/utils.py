"""
Utility functions for validation and data processing.
"""

import re
from typing import Optional, Dict, Any
from pathlib import Path
import json

from openacme.icd10.generate_embeddings import EMBEDDINGS_BASE


def validate_icd10_code(code: str) -> bool:
    """Validate ICD-10 code format.

    Parameters
    ----------
    code : str
        ICD-10 code string (e.g., "I50.9", "A00", "B20.1").

    Returns
    -------
    bool
        True if valid format, False otherwise.
    """
    if not code or not isinstance(code, str):
        return False
    # Pattern: Letter followed by 2 digits, optionally followed by . and more digits
    pattern = r'^[A-Z][0-9]{2}(\.[0-9]+)?$'
    return bool(re.match(pattern, code))


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


def get_icd10_name(code: str, definitions_data: Optional[Dict[str, Any]] = None) -> str:
    """Get human-readable name for an ICD-10 code.

    Parameters
    ----------
    code : str
        ICD-10 code.
    definitions_data : dict, optional
        Optional pre-loaded definitions dict. If None, loads definitions
        from default location.

    Returns
    -------
    str
        Code name, or error message if code not found.
    """
    if definitions_data is None:
        definitions_data = load_icd10_definitions()

    if code not in definitions_data:
        return f"Unknown code: {code}"

    return definitions_data[code].get('name', f"Code: {code}")


def combine_text_for_retrieval(disease_name: str, evidence: list) -> str:
    """Combine disease name and evidence into a single text for retrieval.

    Parameters
    ----------
    disease_name : str
        Name of the disease.
    evidence : list
        List of evidence strings.

    Returns
    -------
    str
        Combined text string.
    """
    evidence_text = "\n".join(evidence) if evidence else ""
    if evidence_text:
        return f"{disease_name}\n\n{evidence_text}"
    return disease_name


def validate_extraction_result(result: Dict[str, Any]) -> bool:
    """Validate structure of disease extraction result.

    Parameters
    ----------
    result : dict
        Result dictionary from LLM.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    if not isinstance(result, dict):
        return False
    if 'Diseases' not in result:
        return False
    if not isinstance(result['Diseases'], list):
        return False
    return True


def format_output(diagnoses: list) -> Dict[str, Any]:
    """Format final output in standardized structure.

    Parameters
    ----------
    diagnoses : list
        List of diagnosis dictionaries.

    Returns
    -------
    dict
        Formatted output dictionary.
    """
    formatted = {
        "diagnoses": [],
        "summary": {
            "total_diagnoses": len(diagnoses),
            "total_codes_retrieved": sum(
                len(d.get('retrieved_codes', [])) for d in diagnoses
            )
        }
    }

    for diag in diagnoses:
        formatted_diag = {
            "disease": diag.get('Disease', ''),
            "evidence": diag.get('Supporting Evidence', []),
            "llm_prediction": {
                "code": diag.get('ICD10', ''),
                "name": diag.get('llm_code_name', ''),
            },
            "retrieved_codes": diag.get('retrieved_codes', []),
            "reranked_codes": diag.get('reranked_codes', []),
        }

        # Add final code (top reranked code)
        if diag.get('reranked_codes'):
            formatted_diag["final_code"] = diag['reranked_codes'][0].get('ICD-10 Code', '')
        elif diag.get('ICD10'):
            formatted_diag["final_code"] = diag['ICD10']
        else:
            formatted_diag["final_code"] = None

        formatted["diagnoses"].append(formatted_diag)

    return formatted

