"""
Utility functions for RAG grounder core.

Includes evidence span finding utilities.
"""
import re
from typing import List

from difflib import SequenceMatcher


def _similarity_ratio(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two strings using built-in difflib."""
    return SequenceMatcher(None, s1, s2).ratio()


def find_evidence_spans(
    text: str,
    evidence_strings: List[str],
    min_similarity: float = 0.7,
    case_sensitive: bool = False,
) -> List[tuple]:
    """Find character spans for evidence strings in text using fuzzy matching.

    Two-pass algorithm: first finds all exact matches, then all fuzzy matches.
    Uses global matched ranges tracker to prevent overlaps across evidence strings.

    Parameters
    ----------
    text : str
        Original text to search in.
    evidence_strings : list of str
        List of evidence strings to find.
    min_similarity : float
        Minimum similarity threshold (0.0 to 1.0) for fuzzy matching.
        Defaults to 0.7.
    case_sensitive : bool
        Whether to preserve case in matching. Defaults to False.

    Returns
    -------
    list of tuple
        List of tuples (start, end, match_type, matched_text, similarity).
        Evidence strings with no matches are discarded.
    """
    if not text or not evidence_strings:
        return []

    # Clean and normalize evidence strings
    evidence_data = []
    for evidence in evidence_strings:
        if not evidence or not evidence.strip():
            continue
        evidence_clean = evidence.strip()
        evidence_normalized = evidence_clean if case_sensitive else evidence_clean.lower()
        evidence_data.append((evidence_clean, evidence_normalized))

    if not evidence_data:
        return []

    text_to_search = text if case_sensitive else text.lower()
    matched_ranges = []
    result = []

    def _is_overlapping(start, end):
        """Check if (start, end) overlaps with any matched range."""
        for r_start, r_end in matched_ranges:
            if not (end <= r_start or start >= r_end):
                return True
        return False

    # Pass 1: Find all exact matches
    exact_results = {}
    for idx, (evidence_clean, evidence_normalized) in enumerate(evidence_data):
        search_start = 0
        while True:
            start_idx = text_to_search.find(evidence_normalized, search_start)
            if start_idx == -1:
                break

            end_idx = start_idx + len(evidence_normalized)
            if not _is_overlapping(start_idx, end_idx):
                matched_text = text[start_idx:end_idx]
                exact_results[idx] = (start_idx, end_idx, matched_text)
                matched_ranges.append((start_idx, end_idx))
                break
            search_start = start_idx + 1

    # Pass 2: Find all fuzzy matches
    fuzzy_results = {}
    words = list(re.finditer(r"\S+", text))
    word_list = [(m.group(), m.start(), m.end()) for m in words]

    if word_list:
        for idx, (evidence_clean, evidence_normalized) in enumerate(evidence_data):
            if idx in exact_results:
                continue

            evidence_word_count = len(evidence_normalized.split())
            best_match = None
            best_similarity = 0.0
            best_start = None
            best_end = None

            for window_size in range(
                evidence_word_count,
                min(evidence_word_count + 5, len(word_list) + 1),
            ):
                for i in range(len(word_list) - window_size + 1):
                    window_words = word_list[i : i + window_size]
                    window_start = window_words[0][1]
                    window_end = window_words[-1][2]

                    if _is_overlapping(window_start, window_end):
                        continue

                    window_text = text[window_start:window_end]
                    window_normalized = (
                        window_text if case_sensitive else window_text.lower()
                    )
                    similarity = _similarity_ratio(evidence_normalized, window_normalized)

                    if similarity > best_similarity and similarity >= min_similarity:
                        best_similarity = similarity
                        best_start = window_start
                        best_end = window_end
                        best_match = window_text

            if best_match:
                fuzzy_results[idx] = (best_start, best_end, best_match, best_similarity)
                matched_ranges.append((best_start, best_end))

    # Build result list
    for idx, (evidence_clean, evidence_normalized) in enumerate(evidence_data):
        if idx in exact_results:
            start, end, matched_text = exact_results[idx]
            result.append((start, end, "exact", matched_text, 1.0))
        elif idx in fuzzy_results:
            start, end, matched_text, similarity = fuzzy_results[idx]
            result.append((start, end, "fuzzy", matched_text, similarity))

    return result
