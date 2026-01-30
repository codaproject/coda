"""
Mention annotation using fuzzy string matching.

Finds character spans of mention strings in the original clinical text
and formats ICD-10 candidate codes into a unified AnnotatedText output.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from difflib import SequenceMatcher
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Pydantic models for structured annotation output format

class RankedMatch(BaseModel):
    """A single ranked match for an annotation."""

    identifier: str = Field(
        ...,
        description="CURIE identifier for the matched entity (e.g., 'hp:0001945', 'icd10:P22.0')",
    )
    name: str = Field(..., description="Human-readable name of the matched entity")
    score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Similarity/confidence score between 0.0 and 1.0, or None if not available",
    )
    properties: Optional[Dict[str, Any]] = Field(
        None, description="Additional properties/metadata for the match"
    )


class Annotation(BaseModel):
    """A single annotation with ranked matches."""

    text: str = Field(..., description="The text span that was annotated")
    top_identifier: str = Field(..., description="CURIE identifier of the top-ranked match")
    top_name: str = Field(..., description="Name of the top-ranked match")
    ranked_matches: List[RankedMatch] = Field(
        ..., description="List of all ranked matches, ordered by score (highest first)"
    )
    properties: Optional[Dict[str, Any]] = Field(
        None, description="Additional properties/metadata for the annotation"
    )


class AnnotatedText(BaseModel):
    """Complete annotated text output."""

    text: str = Field(..., description="The original input text")
    type: str = Field(..., description="Type of text (e.g., 'va_narrative', 'clinical_note')")
    identifier: str = Field(
        ...,
        description="Unique identifier for this text (e.g., 'champs_deid:A31C4B5E-3890-4AEC-8375-37DB6D916AED')",
    )
    annotations: List[Annotation] = Field(
        default_factory=list, description="List of annotations found in the text"
    )
    properties: Optional[Dict[str, Any]] = Field(
        None, description="Additional properties/metadata for the annotated text (e.g., model, provider)"
    )


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

    Returns exactly one match per evidence string (1:1 correspondence).
    For duplicate mentions, finds non-overlapping occurrences.
    Prioritizes exact matches over fuzzy matches.

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
        List of matches, one per input evidence string. Each dict has keys:
        - text: matched text span
        - start: start position (None if not_found)
        - end: end position (None if not_found)
        - similarity: similarity score (0.0-1.0)
        - match_type: "exact", "fuzzy", or "not_found"
    """
    if not clinical_text or not evidence_strings:
        return [{"text": "", "start": None, "end": None, "similarity": 0.0, "match_type": "not_found"}] * len(evidence_strings) if evidence_strings else []

    annotated_evidence: List[Dict[str, Any]] = []
    # Track assigned positions to avoid overlaps for duplicate mentions
    assigned_positions: List[tuple[int, int]] = []  # List of (start, end) tuples
    
    def _spans_overlap(start1: int, end1: int, start2: int, end2: int) -> bool:
        """Check if two spans overlap (adjacent spans don't overlap)."""
        return not (end1 <= start2 or start1 >= end2)
    
    def _find_all_matches_for_evidence(evidence: str) -> List[Dict[str, Any]]:
        """Find all possible matches (exact + fuzzy) for a single evidence string."""
        if not evidence or not evidence.strip():
            return []
        
        evidence_clean = evidence.strip()
        evidence_normalized = evidence_clean if case_sensitive else evidence_clean.lower()
        evidence_words = evidence_normalized.split()
        
        if not evidence_words:
            return []
        
        all_matches: List[Dict[str, Any]] = []
        
        # Build regex pattern for exact matching
        escaped_words = [re.escape(word) for word in evidence_words]
        pattern = r'\s+'.join(escaped_words)
        flags = 0 if case_sensitive else re.IGNORECASE
        
        # Find all exact matches
        exact_matches = list(re.finditer(pattern, clinical_text, flags))
        for match in exact_matches:
            start_idx = match.start()
            end_idx = match.end()
            matched_text = clinical_text[start_idx:end_idx]
            all_matches.append({
                "text": matched_text,
                "start": start_idx,
                "end": end_idx,
                "similarity": 1.0,
                "match_type": "exact",
            })
        
        # If we found exact matches, return them (prioritize exact over fuzzy)
        if all_matches:
            return all_matches
        
        # Find fuzzy matches
        words = re.finditer(r"\S+", clinical_text)
        word_list = [(m.group(), m.start(), m.end()) for m in words]
        
        if not word_list:
            return []

        evidence_word_count = len(evidence_normalized.split())
        fuzzy_matches: List[Dict[str, Any]] = []

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

                if similarity >= min_similarity:
                    fuzzy_matches.append({
                        "text": window_text,
                        "start": window_start_char,
                        "end": window_end_char,
                        "similarity": similarity,
                        "match_type": "fuzzy",
                    })
        
        return fuzzy_matches
    
    # Process each evidence string in order
    for evidence in evidence_strings:
        if not evidence or not evidence.strip():
            annotated_evidence.append({
                "text": "",
                "start": None,
                "end": None,
                "similarity": 0.0,
                "match_type": "not_found",
            })
            continue
        
        evidence_clean = evidence.strip()
        
        # Find all possible matches for this evidence string
        all_matches = _find_all_matches_for_evidence(evidence)
        
        if not all_matches:
            annotated_evidence.append({
                "text": evidence_clean,
                "start": None,
                "end": None,
                "similarity": 0.0,
                "match_type": "not_found",
            })
            continue
        
        # Sort matches: exact matches first, then by similarity (highest first), then by position (leftmost first)
        all_matches.sort(key=lambda x: (
            0 if x["match_type"] == "exact" else 1,  # Exact matches first
            -x["similarity"],  # Higher similarity first
            x["start"]  # Leftmost first (tie-breaker)
        ))
        
        # Find the best non-overlapping match
        best_match = None
        for match in all_matches:
            start = match["start"]
            end = match["end"]
            
            # Check if this match overlaps with any already-assigned position
            overlaps = False
            for assigned_start, assigned_end in assigned_positions:
                if _spans_overlap(start, end, assigned_start, assigned_end):
                    overlaps = True
                    break
            
            if not overlaps:
                best_match = match
                assigned_positions.append((start, end))
                break
        
        # If no non-overlapping match found, return "not_found"
        if best_match is None:
            annotated_evidence.append({
                'text': evidence_clean,
                'start': None,
                'end': None,
                'similarity': 0.0,
                'match_type': 'not_found'
            })
        else:
            annotated_evidence.append(best_match)
    
    return annotated_evidence


def annotate(
    text: str,
    text_type: str,
    identifier: str,
    pipeline_output: Dict[str, Any],
    min_similarity: float = 0.7,
    case_sensitive: bool = False,
    properties: Optional[Dict[str, Any]] = None,
) -> AnnotatedText:
    """
    Annotate text with ICD-10 codes and evidence spans.

    Parameters
    ----------
    text : str
        Original clinical description text.
    text_type : str
        Type of text (e.g., 'va_narrative', 'clinical_note').
    identifier : str
        Unique identifier for this text (e.g., 'champs_deid:A31C4B5E-3890-4AEC-8375-37DB6D916AED').
    pipeline_output : dict
        Raw pipeline result dictionary.
    min_similarity : float
        Minimum similarity threshold (0.0 to 1.0) for fuzzy matching.
        Defaults to 0.7.
    case_sensitive : bool
        Whether to preserve case in matching. Defaults to False.
    """
    # Save the function parameter to avoid shadowing
    annotated_text_properties = properties
    
    if not isinstance(pipeline_output, dict) or "Mentions" not in pipeline_output:
        return AnnotatedText(
            text=text,
            type=text_type,
            identifier=identifier,
            annotations=[],
            properties=annotated_text_properties,
        )

    mentions = pipeline_output.get("Mentions", [])
    annotations: List[Annotation] = []

    for mention_item in mentions:
        mention_text = (mention_item.get("Mention", "") or "").strip()
        if not mention_text:
            continue

        # Find the best match for this mention (find_evidence_spans returns exactly one match per mention)
        spans = find_evidence_spans(
            text,
            [mention_text],
            min_similarity=min_similarity,
            case_sensitive=case_sensitive,
        )
        
        # find_evidence_spans returns a list with exactly one element (or empty if input was empty)
        if not spans:
            logger.warning(
                f"Mention '{mention_text[:60]}...' returned no spans from find_evidence_spans. "
                f"This should not happen."
            )
            continue
        
        span = spans[0]  # Get the single match for this mention
        
        # Check if match is valid (shouldn't happen due to validation, but be safe)
        if span.get("match_type") == "not_found":
            logger.warning(
                f"Mention '{mention_text[:60]}...' could not be matched in text. "
                f"This should have been filtered out during validation in process_raw()."
            )
            continue

        # Get codes for this mention
        reranked_codes = mention_item.get("reranked_codes", [])
        retrieved_codes = mention_item.get("retrieved_codes", [])
        reranking_failed = bool(mention_item.get("reranking_failed", False))

        code_to_definition: Dict[str, str] = {}
        for retrieved in retrieved_codes:
            code = retrieved.get("code", "")
            definition = retrieved.get("definition", "")
            if code and definition:
                code_to_definition[code] = definition

        use_retrieved_fallback = reranking_failed or not reranked_codes

        if use_retrieved_fallback:
            if not retrieved_codes:
                continue

            sorted_retrieved = sorted(
                retrieved_codes,
                key=lambda x: x.get("similarity", 0.0),
                reverse=True,
            )

            codes_to_process = [
                {
                    "ICD-10 Code": r.get("code", ""),
                    "ICD-10 Name": r.get("name", ""),
                    "similarity": r.get("similarity", None),
                    "retrieved_from": "semantic_embeddings",
                }
                for r in sorted_retrieved
            ]

            top_code_info = codes_to_process[0]
        else:
            # Use all reranked codes (reranker returns best-to-worst order)
            codes_to_process = reranked_codes
            # Top code is always the first reranked code (best match)
            top_code_info = reranked_codes[0] if reranked_codes else {}

        ranked_matches: List[RankedMatch] = []
        for code_info in codes_to_process:
            code = code_info.get("ICD-10 Code", "")
            name = code_info.get("ICD-10 Name", "")
            similarity = code_info.get("similarity", None)

            properties = {"retrieved_from": code_info.get("retrieved_from", "unknown")}
            if code and code in code_to_definition:
                properties["definition"] = code_to_definition[code]

            ranked_matches.append(
                RankedMatch(
                    identifier=f"icd10:{code}" if code else "",
                    name=name,
                    score=float(similarity) if similarity is not None else None,
                    properties=properties,
                )
            )

        # Create exactly one annotation for this mention
        annotation_text = span["text"]
        span_source = "matched_exact" if span.get("match_type") == "exact" else "matched_fuzzy"
        
        annotation_properties = {
            "reranking_mode": "similarity_based" if use_retrieved_fallback else "llm_reranked",
            "span_source": span_source,
            "span_start": span.get("start"),
            "span_end": span.get("end"),
        }
        
        annotations.append(
            Annotation(
                text=annotation_text,
                top_identifier=f"icd10:{top_code_info.get('ICD-10 Code', '')}",
                top_name=top_code_info.get("ICD-10 Name", ""),
                ranked_matches=ranked_matches,
                properties=annotation_properties,
            )
        )

    return AnnotatedText(
        text=text,
        type=text_type,
        identifier=identifier,
        annotations=annotations,
        properties=annotated_text_properties,
    )
