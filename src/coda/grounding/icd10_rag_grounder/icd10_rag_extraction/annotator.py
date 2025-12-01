"""
Evidence annotation using fuzzy string matching.

Finds character spans of evidence strings in the original clinical text.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from difflib import SequenceMatcher


def _similarity_ratio(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two strings using built-in difflib.

    Parameters
    ----------
    s1 : str
        First string.
    s2 : str
        Second string.

    Returns
    -------
    float
        Similarity ratio between 0.0 and 1.0.
    """
    return SequenceMatcher(None, s1, s2).ratio()


def find_evidence_spans(
    clinical_text: str,
    evidence_strings: List[str],
    min_similarity: float = 0.7,
    case_sensitive: bool = False
) -> List[Dict[str, Any]]:
    """Find character spans for evidence strings in clinical text using fuzzy matching.

    Parameters
    ----------
    clinical_text : str
        Original clinical description text.
    evidence_strings : list of str
        List of evidence strings to find.
    min_similarity : float
        Minimum similarity threshold (0.0 to 1.0) for fuzzy matching.
        Defaults to 0.7.
    case_sensitive : bool
        Whether to preserve case in matching. Defaults to False.

    Returns
    -------
    list of dict
        List of dictionaries with 'text', 'start', 'end', and 'similarity' keys.
    """
    if not clinical_text or not evidence_strings:
        return []

    # Normalize text for matching
    text_to_search = clinical_text if case_sensitive else clinical_text.lower()

    annotated_evidence = []

    for evidence in evidence_strings:
        if not evidence or not evidence.strip():
            continue

        evidence_clean = evidence.strip()
        evidence_normalized = evidence_clean if case_sensitive else evidence_clean.lower()

        # Try exact match first
        exact_match = False
        if case_sensitive:
            start_idx = clinical_text.find(evidence_clean)
        else:
            start_idx = text_to_search.find(evidence_normalized)

        if start_idx != -1:
            end_idx = start_idx + len(evidence_clean)
            annotated_evidence.append({
                'text': evidence_clean,
                'start': start_idx,
                'end': end_idx,
                'similarity': 1.0,
                'match_type': 'exact'
            })
            exact_match = True
            continue

        # If no exact match, try fuzzy matching
        # Use sliding window approach on original text to preserve exact character positions
        words = re.finditer(r'\S+', clinical_text)  # Find word boundaries with positions
        word_list = [(m.group(), m.start(), m.end()) for m in words]

        if not word_list:
            annotated_evidence.append({
                'text': evidence_clean,
                'start': None,
                'end': None,
                'similarity': 0.0,
                'match_type': 'not_found'
            })
            continue

        # Try to find best match using sliding window
        best_match = None
        best_similarity = 0.0

        # Search with different window sizes
        evidence_word_count = len(evidence_normalized.split())
        for window_size in range(evidence_word_count, min(evidence_word_count + 5, len(word_list) + 1)):
            for i in range(len(word_list) - window_size + 1):
                # Get window words and their positions
                window_words = word_list[i:i+window_size]
                window_start_char = window_words[0][1]  # Start of first word
                window_end_char = window_words[-1][2]   # End of last word

                # Extract actual text from original (preserves exact spacing)
                window_text = clinical_text[window_start_char:window_end_char]
                window_normalized = window_text if case_sensitive else window_text.lower()

                # Calculate similarity
                similarity = _similarity_ratio(evidence_normalized, window_normalized)

                if similarity > best_similarity and similarity >= min_similarity:
                    best_similarity = similarity
                    best_match = {
                        'text': window_text,
                        'start': window_start_char,
                        'end': window_end_char,
                        'similarity': similarity,
                        'match_type': 'fuzzy'
                    }

        if best_match:
            annotated_evidence.append(best_match)
        else:
            # No match found - add with null span
            annotated_evidence.append({
                'text': evidence_clean,
                'start': None,
                'end': None,
                'similarity': 0.0,
                'match_type': 'not_found'
            })

    return annotated_evidence


def annotate_pipeline_output(
    clinical_description: str,
    pipeline_output: Dict[str, Any],
    min_similarity: float = 0.7,
    case_sensitive: bool = False
) -> Dict[str, Any]:
    """Add character spans to evidence in pipeline output.

    Parameters
    ----------
    clinical_description : str
        Original clinical description text.
    pipeline_output : dict
        Output from MedCoderPipeline.process().
    min_similarity : float
        Minimum similarity threshold for fuzzy matching. Defaults to 0.7.
    case_sensitive : bool
        Whether to preserve case in matching. Defaults to False.

    Returns
    -------
    dict
        Pipeline output with added 'evidence_spans' field for each diagnosis.
    """
    if not isinstance(pipeline_output, dict) or 'diagnoses' not in pipeline_output:
        return pipeline_output

    annotated_output = pipeline_output.copy()
    annotated_output['diagnoses'] = []

    for diagnosis in pipeline_output['diagnoses']:
        annotated_diagnosis = diagnosis.copy()

        # Get evidence strings
        evidence = diagnosis.get('evidence', [])

        if evidence:
            # Find spans for each evidence string
            evidence_spans = find_evidence_spans(
                clinical_description,
                evidence,
                min_similarity=min_similarity,
                case_sensitive=case_sensitive
            )
            annotated_diagnosis['evidence_spans'] = evidence_spans
        else:
            annotated_diagnosis['evidence_spans'] = []

        annotated_output['diagnoses'].append(annotated_diagnosis)

    return annotated_output


def annotate_raw_output(
    clinical_description: str,
    raw_output: Dict[str, Any],
    min_similarity: float = 0.7,
    case_sensitive: bool = False
) -> Dict[str, Any]:
    """Add character spans to evidence in raw pipeline output (non-formatted).

    Parameters
    ----------
    clinical_description : str
        Original clinical description text.
    raw_output : dict
        Raw output from MedCoderPipeline.process().
    min_similarity : float
        Minimum similarity threshold for fuzzy matching. Defaults to 0.7.
    case_sensitive : bool
        Whether to preserve case in matching. Defaults to False.

    Returns
    -------
    dict
        Raw output with added 'evidence_spans' field for each disease.
    """
    if not isinstance(raw_output, dict) or 'Diseases' not in raw_output:
        return raw_output

    annotated_output = raw_output.copy()
    annotated_output['Diseases'] = []

    for disease in raw_output['Diseases']:
        annotated_disease = disease.copy()

        # Get evidence strings
        evidence = disease.get('Supporting Evidence', [])

        if evidence:
            # Find spans for each evidence string
            evidence_spans = find_evidence_spans(
                clinical_description,
                evidence,
                min_similarity=min_similarity,
                case_sensitive=case_sensitive
            )
            annotated_disease['evidence_spans'] = evidence_spans
        else:
            annotated_disease['evidence_spans'] = []

        annotated_output['Diseases'].append(annotated_disease)

    return annotated_output

