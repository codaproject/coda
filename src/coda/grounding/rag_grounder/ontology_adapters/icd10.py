"""
ICD-10 ontology adapter for RAG retrieval terms.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from openacme.icd10 import get_icd10_graph

from ..retrieval_term import RetrievalTerm


def _to_string_list(value: Any) -> list[str]:
    """Normalize unknown values into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Iterable):
        values = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                values.append(text)
        return values
    text = str(value).strip()
    return [text] if text else []


def _extract_rubrics(data: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return normalized rubric lists keyed by rubric type."""
    raw_rubrics = data.get("rubrics", {})
    if not isinstance(raw_rubrics, Mapping):
        return {}

    rubrics: dict[str, list[str]] = {}
    for key, value in raw_rubrics.items():
        normalized = _to_string_list(value)
        if normalized:
            rubrics[str(key)] = normalized
    return rubrics


def load_icd10_retrieval_terms(prefix: str = "icd10") -> list[RetrievalTerm]:
    """
    Load ICD-10 terms as RetrievalTerm objects for RAG indexing.

    Parameters
    ----------
    prefix : str
        CURIE namespace prefix used in returned term IDs.

    Returns
    -------
    list[RetrievalTerm]
        Deterministically ordered retrieval terms.
    """
    graph = get_icd10_graph()
    terms: list[RetrievalTerm] = []

    for code, data in graph.nodes(data=True):
        data_mapping = data if isinstance(data, Mapping) else {}
        rubrics = _extract_rubrics(data_mapping)

        preferred = rubrics.get("preferred", [])
        name = preferred[0] if preferred else str(code)

        synonyms: list[str] = []
        for rubric_key, rubric_values in rubrics.items():
            if rubric_key == "preferred":
                continue
            synonyms.extend(rubric_values)
        # Deduplicate while preserving order.
        synonyms = list(dict.fromkeys(synonyms))

        definition = synonyms[0] if synonyms else None

        terms.append(
            RetrievalTerm(
                id=f"{prefix}:{code}",
                name=name,
                definition=definition,
                synonyms=synonyms or None,
            )
        )

    terms.sort(key=lambda term: term.id)
    return terms
