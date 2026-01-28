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
    """Calculate similarity ratio between two strings using built-in difflib."""
    return SequenceMatcher(None, s1, s2).ratio()


def find_evidence_spans(
    clinical_text: str,
    evidence_strings: List[str],
    min_similarity: float = 0.7,
    case_sensitive: bool = False,
) -> List[Dict[str, Any]]:
    """Find character spans for evidence strings in clinical text using fuzzy matching."""
    if not clinical_text or not evidence_strings:
        return []

    text_to_search = clinical_text if case_sensitive else clinical_text.lower()
    annotated_evidence: List[Dict[str, Any]] = []

    for evidence in evidence_strings:
        if not evidence or not evidence.strip():
            continue

        evidence_clean = evidence.strip()
        evidence_normalized = evidence_clean if case_sensitive else evidence_clean.lower()
        
        # Normalize whitespace for matching: collapse multiple spaces to single space
        evidence_words = evidence_normalized.split()
        if not evidence_words:
            continue
        
        # Build regex pattern that matches words with flexible whitespace
        # Escape special regex characters in words
        escaped_words = [re.escape(word) for word in evidence_words]
        pattern = r'\s+'.join(escaped_words)
        
        # Try exact match first (with flexible whitespace)
        flags = 0 if case_sensitive else re.IGNORECASE
        match = re.search(pattern, clinical_text, flags)
        if match:
            start_idx = match.start()
            end_idx = match.end()
            matched_text = clinical_text[start_idx:end_idx]
            annotated_evidence.append(
                {
                    "text": matched_text,
                    "start": start_idx,
                    "end": end_idx,
                    "similarity": 1.0,
                    "match_type": "exact",
                }
            )
            continue

        # Fuzzy match with sliding window across word spans
        words = re.finditer(r"\S+", clinical_text)
        word_list = [(m.group(), m.start(), m.end()) for m in words]

        if not word_list:
            annotated_evidence.append(
                {
                    "text": evidence_clean,
                    "start": None,
                    "end": None,
                    "similarity": 0.0,
                    "match_type": "not_found",
                }
            )
            continue

        best_match = None
        best_similarity = 0.0

        evidence_word_count = len(evidence_normalized.split())
        for window_size in range(
            evidence_word_count, min(evidence_word_count + 5, len(word_list) + 1)
        ):
            for i in range(len(word_list) - window_size + 1):
                window_words = word_list[i : i + window_size]
                window_start_char = window_words[0][1]
                window_end_char = window_words[-1][2]

                window_text = clinical_text[window_start_char:window_end_char]
                window_normalized = window_text if case_sensitive else window_text.lower()

                similarity = _similarity_ratio(evidence_normalized, window_normalized)

                if similarity > best_similarity and similarity >= min_similarity:
                    best_similarity = similarity
                    best_match = {
                        "text": window_text,
                        "start": window_start_char,
                        "end": window_end_char,
                        "similarity": similarity,
                        "match_type": "fuzzy",
                    }

        if best_match:
            annotated_evidence.append(best_match)
        else:
            annotated_evidence.append(
                {
                    "text": evidence_clean,
                    "start": None,
                    "end": None,
                    "similarity": 0.0,
                    "match_type": "not_found",
                }
            )

    return annotated_evidence


def annotate(
    text: str,
    text_type: str,
    identifier: str,
    pipeline_result: Dict[str, Any],
    add_evidence_spans: bool = False,
    min_similarity: float = 0.7,
    case_sensitive: bool = False,
    properties: Optional[Dict[str, Any]] = None,
) -> AnnotatedText:
    """
    Contract B:
    - Annotation.text is anchored to the original input text whenever possible
      (exact/fuzzy matched substring).
    - LLM Mention is treated as a proposal; fallback only if no span can be found.
    - span provenance is exposed via Annotation.properties["span_source"].

    Parameters
    ----------
    properties : Dict[str, Any], optional
        Additional properties/metadata for the annotated text.
        Common properties: model, provider, etc.
    """
    # Save the function parameter to avoid shadowing
    annotated_text_properties = properties
    
    if not isinstance(pipeline_result, dict) or "Mentions" not in pipeline_result:
        return AnnotatedText(
            text=text,
            type=text_type,
            identifier=identifier,
            annotations=[],
            properties=annotated_text_properties,
        )

    mentions = pipeline_result.get("Mentions", [])
    annotations: List[Annotation] = []

    for mention_item in mentions:
        mention_text = (mention_item.get("Mention", "") or "").strip()
        if not mention_text:
            continue

        # Try to anchor the mention to the original text
        spans = find_evidence_spans(
            text,
            [mention_text],
            min_similarity=min_similarity,
            case_sensitive=case_sensitive,
        )
        best_span = spans[0] if spans else None

        if best_span and best_span.get("match_type") == "exact":
            annotation_text = best_span["text"]
            span_source = "matched_exact"
        elif best_span and best_span.get("match_type") == "fuzzy":
            annotation_text = best_span["text"]
            span_source = "matched_fuzzy"
        else:
            # This should not happen - mentions are validated in pipeline.process_raw()
            # before reaching annotate(), so all mentions should be matchable.
            # If this occurs, it indicates a bug in the validation logic.
            raise ValueError(
                f"Mention '{mention_text[:60]}...' could not be matched in text "
                f"(match_type: {best_span.get('match_type') if best_span else 'not_found'}). "
                f"This should have been filtered out during validation in process_raw()."
            )

        if add_evidence_spans:
            mention_item["mention_spans"] = spans

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

        annotations.append(
            Annotation(
                text=annotation_text,
                top_identifier=f"icd10:{top_code_info.get('ICD-10 Code', '')}",
                top_name=top_code_info.get("ICD-10 Name", ""),
                ranked_matches=ranked_matches,
                properties={
                    "reranking_mode": "similarity_based" if use_retrieved_fallback else "llm_reranked",
                    "span_source": span_source,
                },
            )
        )

    return AnnotatedText(
        text=text,
        type=text_type,
        identifier=identifier,
        annotations=annotations,
        properties=annotated_text_properties,
    )
