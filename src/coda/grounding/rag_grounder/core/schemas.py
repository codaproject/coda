"""
JSON schemas for structured LLM outputs in generalized RAG grounder.

Schemas are parametrized by concept_type (e.g., "disease", "vaccine", "medication").
"""
from typing import Any, Dict, List

from pydantic import BaseModel


def get_extraction_schema(concept_type: str) -> Dict[str, Any]:
    """
    Get extraction schema parametrized by concept type.

    Parameters
    ----------
    concept_type : str
        Type of concept to extract (e.g., "disease", "vaccine", "medication").

    Returns
    -------
    Dict[str, Any]
        JSON schema for concept extraction.
    """
    concept_type_capitalized = concept_type.capitalize()

    return {
        "type": "object",
        "properties": {
            "Concepts": {
                "type": "array",
                "description": f"The {concept_type_capitalized} concepts extracted from the text.",
                "items": {
                    "type": "object",
                    "properties": {
                        "Concept": {
                            "type": "string",
                            "description": f"The {concept_type_capitalized} concept name extracted from the text.",
                        },
                        "Supporting_Evidence": {
                            "type": "array",
                            "description": "Exact verbatim text spans from the input text that support the concept. DO NOT paraphrase or reword. Extract the exact text as it appears in the input.",
                            "items": {
                                "type": "string",
                                "description": "A verbatim text span copied exactly from the input text. Must be an exact substring of the input text, not a paraphrase or summary.",
                            },
                        },
                    },
                    "required": ["Concept", "Supporting_Evidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["Concepts"],
        "additionalProperties": False,
    }


RERANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "Reranked_Terms": {
            "type": "array",
            "description": "The re-ranked terms, ordered from most to least appropriate.",
            "items": {
                "type": "object",
                "properties": {
                    "Term_Identifier": {
                        "type": "string",
                        "description": "The identifier of the term.",
                    },
                    "Term_Name": {
                        "type": "string",
                        "description": "The human-readable name of the term.",
                    },
                },
                "required": ["Term_Identifier", "Term_Name"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["Reranked_Terms"],
    "additionalProperties": False,
}


BATCH_RERANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "Reranked_Terms_Batch": {
            "type": "array",
            "description": "Array of reranked term lists, one per concept. Each object contains an index and the reranked terms for that concept.",
            "items": {
                "type": "object",
                "properties": {
                    "Concept_Index": {
                        "type": "integer",
                        "description": "The index of the concept (0-based) that this reranked list corresponds to.",
                    },
                    "Reranked_Terms": {
                        "type": "array",
                        "description": "Reranked terms for this concept, ordered from most to least appropriate.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Term_Identifier": {
                                    "type": "string",
                                    "description": "The identifier of the term.",
                                },
                                "Term_Name": {
                                    "type": "string",
                                    "description": "The human-readable name of the term.",
                                },
                            },
                            "required": ["Term_Identifier", "Term_Name"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["Concept_Index", "Reranked_Terms"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["Reranked_Terms_Batch"],
    "additionalProperties": False,
}


# Pydantic models for validation
class ConceptItem(BaseModel):
    """Pydantic model for a single concept."""

    Concept: str
    Supporting_Evidence: List[str]


class ExtractionResult(BaseModel):
    """Pydantic model for extraction result."""

    Concepts: List[ConceptItem]


class RerankedTermItem(BaseModel):
    """Pydantic model for a single reranked term."""

    Term_Identifier: str
    Term_Name: str


class RerankingResult(BaseModel):
    """Pydantic model for reranking result."""

    Reranked_Terms: List[RerankedTermItem]


class ConceptRerankingItem(BaseModel):
    """Pydantic model for a single concept's reranking result."""

    Concept_Index: int
    Reranked_Terms: List[RerankedTermItem]


class BatchRerankingResult(BaseModel):
    """Pydantic model for batch reranking result."""

    Reranked_Terms_Batch: List[ConceptRerankingItem]
