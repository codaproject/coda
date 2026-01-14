"""
Evidence annotation using fuzzy string matching.

Finds character spans of evidence strings in the original clinical text.
"""

import re
from typing import Dict, Any, List
from difflib import SequenceMatcher
from pydantic import BaseModel, Field


# Pydantic models for structured annotation output format

class RankedMatch(BaseModel):
    """A single ranked match for an annotation."""
    
    identifier: str = Field(..., description="CURIE identifier for the matched entity (e.g., 'hp:0001945', 'icd10:P22.0')")
    name: str = Field(..., description="Human-readable name of the matched entity")
    score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity/confidence score between 0.0 and 1.0, or None if not available")
    properties: Optional[Dict[str, Any]] = Field(None, description="Additional properties/metadata for the match")


class Annotation(BaseModel):
    """A single annotation with ranked matches."""
    
    text: str = Field(..., description="The text span that was annotated")
    top_identifier: str = Field(..., description="CURIE identifier of the top-ranked match")
    top_name: str = Field(..., description="Name of the top-ranked match")
    ranked_matches: List[RankedMatch] = Field(..., description="List of all ranked matches, ordered by score (highest first)")


class AnnotatedText(BaseModel):
    """Complete annotated text output."""
    
    text: str = Field(..., description="The original input text")
    type: str = Field(..., description="Type of text (e.g., 'va_narrative', 'clinical_note')")
    identifier: str = Field(..., description="Unique identifier for this text (e.g., 'champs_deid:A31C4B5E-3890-4AEC-8375-37DB6D916AED')")
    annotations: List[Annotation] = Field(default_factory=list, description="List of annotations found in the text")


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


def annotate(
    text: str,
    text_type: str,
    identifier: str,
    pipeline_result: Dict[str, Any],
    top_k: int = 5,
    add_evidence_spans: bool = False,
    min_similarity: float = 0.7,
    case_sensitive: bool = False
) -> AnnotatedText:
    """Convert pipeline output to AnnotatedText format, optionally adding evidence spans.
    
    Parameters
    ----------
    text : str
        Original input text.
    text_type : str
        Type of text (e.g., 'va_narrative', 'clinical_note').
    identifier : str
        Unique identifier for the text.
    pipeline_result : dict
        Pipeline output with 'Diseases' key.
    top_k : int
        Number of top ranked matches to include per annotation. Defaults to 5.
    add_evidence_spans : bool
        If True, add character spans for evidence strings. Defaults to False.
    min_similarity : float
        Minimum similarity threshold for evidence span matching (0.0-1.0). Defaults to 0.7.
    case_sensitive : bool
        Whether to preserve case in evidence matching. Defaults to False.
        
    Returns
    -------
    AnnotatedText
        Structured annotation output.
    """
    if not isinstance(pipeline_result, dict) or 'Diseases' not in pipeline_result:
        return AnnotatedText(text=text, type=text_type, identifier=identifier, annotations=[])
    
    # Optionally add evidence spans to the raw result
    if add_evidence_spans:
        pipeline_result = pipeline_result.copy()
        annotated_diseases = []
        
        for disease in pipeline_result.get('Diseases', []):
            annotated_disease = disease.copy()
            evidence = disease.get('Supporting Evidence', [])
            
            if evidence:
                evidence_spans = find_evidence_spans(
                    text,
                    evidence,
                    min_similarity=min_similarity,
                    case_sensitive=case_sensitive
                )
                annotated_disease['evidence_spans'] = evidence_spans
            else:
                annotated_disease['evidence_spans'] = []
            
            annotated_diseases.append(annotated_disease)
        
        pipeline_result['Diseases'] = annotated_diseases
    
    diseases = pipeline_result.get('Diseases', [])
    annotations = []
    
    for disease in diseases:
        disease_name = disease.get('Disease', '')
        evidence = disease.get('Supporting Evidence', [])
        reranked_codes = disease.get('reranked_codes', [])
        retrieved_codes = disease.get('retrieved_codes', [])
        
        if not reranked_codes:
            continue
        
        # Create a lookup map from retrieved codes for definitions
        code_to_definition = {}
        for retrieved in retrieved_codes:
            code = retrieved.get('code', '')
            definition = retrieved.get('definition', '')
            if code and definition:
                code_to_definition[code] = definition
            
        # Get top code (first in reranked list)
        top_code_info = reranked_codes[0]
        top_code = top_code_info.get('ICD-10 Code', '')
        top_name = top_code_info.get('ICD-10 Name', '')
        
        # Convert reranked codes to ranked matches (limit to top_k)
        ranked_matches = []
        for code_info in reranked_codes[:top_k]:
            code = code_info.get('ICD-10 Code', '')
            name = code_info.get('ICD-10 Name', '')
            similarity = code_info.get('similarity', None)
            retrieved_from = code_info.get('retrieved_from', 'unknown')
            
            # Format identifier as CURIE (icd10:CODE)
            curie_identifier = f"icd10:{code}" if code else ""
            
            # Build properties dict with definition (if available) and retrieved_from
            properties = {}
            if code and code in code_to_definition:
                properties["definition"] = code_to_definition[code]
            properties["retrieved_from"] = retrieved_from
            
            # Only include properties if it has at least one field
            if not properties:
                properties = None
            
            ranked_matches.append(
                RankedMatch(
                    identifier=curie_identifier,
                    name=name,
                    score=float(similarity) if similarity is not None else None,
                    properties=properties
                )
            )
        
        # Use first evidence span as annotation text, fallback to disease name if no evidence
        annotation_text = evidence[0] if evidence else disease_name
        
        # Create annotation - use evidence span as the annotation text
        annotation = Annotation(
            text=annotation_text,
            top_identifier=f"icd10:{top_code}" if top_code else "",
            top_name=top_name,
            ranked_matches=ranked_matches
        )
        annotations.append(annotation)
    
    result = AnnotatedText(
        text=text,
        type=text_type,
        identifier=identifier,
        annotations=annotations
    )
    
    # Store raw result for internal use (e.g., evidence spans)
    result._raw_result = pipeline_result
    
    return result

