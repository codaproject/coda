"""
JSON schemas for structured LLM outputs.
"""

# -------------------------------------------------------------------
# Disease / condition grouped extraction (preferred for clean evidence)
# -------------------------------------------------------------------
DISEASE_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "Diseases": {
            "type": "array",
            "description": "The diseases or conditions that the patient likely has.",
            "items": {
                "type": "object",
                "properties": {
                    "Disease": {
                        "type": "string",
                        "description": "The likely disease or condition that the patient has based on the diagnostic description."
                    },
                    "Supporting Evidence": {
                        "type": "array",
                        "description": "Exact verbatim text spans from the clinical description that support the disease. DO NOT paraphrase or reword. Extract the exact text as it appears in the input.",
                        "items": {
                            "type": "string",
                            "description": "A verbatim text span copied exactly from the clinical description. Must be an exact substring of the input text, not a paraphrase or summary."
                        }
                    },
                },
                "required": ["Disease", "Supporting Evidence"],
                "additionalProperties": False
            }
        }
    },
    "required": ["Diseases"],
    "additionalProperties": False
}

# -------------------------------------------------------------------
# General flat evidence span extraction (non-hierarchical, for general-purpose span extraction)
# -------------------------------------------------------------------
COD_EVIDENCE_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "COD_EVIDENCE_SPANS": {
            "type": "array",
            "description": (
                "Clinical evidence spans extracted from the text. "
                "Each item should be a verbatim text span describing a disease, disorder, complication, "
                "injury, abnormal finding, or clinical condition."
            ),
            "items": {
                "type": "string",
                "description": (
                    "An EXACT verbatim text span copied from the input text that expresses a clinical condition. "
                    "DO NOT paraphrase or reword. Extract the exact text as it appears in the input. "
                    "Must be an exact substring of the input text, not a summary or interpretation."
                ),
            },
        }
    },
    "required": ["COD_EVIDENCE_SPANS"],
    "additionalProperties": False,
}

# -------------------------------------------------------------------
# Single-mention reranking schema (legacy)
# -------------------------------------------------------------------
RERANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "Reranked ICD-10 Codes": {
            "type": "array",
            "description": "The re-ranked ICD-10 codes, ordered from most to least appropriate.",
            "items": {
                "type": "object",
                "properties": {
                    "ICD-10 Code": {
                        "type": "string",
                        "description": "The ICD-10 code.",
                    },
                    "ICD-10 Name": {
                        "type": "string",
                        "description": "The human-readable name corresponding to the ICD-10 code.",
                    },
                },
                "required": ["ICD-10 Code", "ICD-10 Name"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["Reranked ICD-10 Codes"],
    "additionalProperties": False,
}

# -------------------------------------------------------------------
# Batch reranking schema (one LLM call for all mentions)
# Simplified: only codes required (names looked up from retrieved_codes)
# -------------------------------------------------------------------
BATCH_RERANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "Mention Rerankings": {
            "type": "array",
            "description": (
                "A list of reranking results, one per mention_id provided in the prompt."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "mention_id": {
                        "type": "integer",
                        "description": "The exact integer 'id' value of the mention.",
                    },
                    "Reranked ICD-10 Codes": {
                        "type": "array",
                        "description": (
                            "ICD-10 codes ordered from most to least appropriate for this mention. "
                            "Return only the codes (strings), not names. Names will be looked up automatically."
                        ),
                        "items": {
                            "type": "string",
                            "description": "An ICD-10 code from the candidate list for this mention.",
                        },
                    },
                },
                "required": ["mention_id", "Reranked ICD-10 Codes"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["Mention Rerankings"],
    "additionalProperties": False,
}

# -------------------------------------------------------------------
# Type hints for better IDE support
# -------------------------------------------------------------------
from typing import Dict, Any, List

MentionInfo = Dict[str, Any]              # {"Mention": str, "ICD10": str}
MentionExtractionResult = Dict[str, Any]  # {"Mentions": List[MentionInfo]}

RerankedCode = Dict[str, Any]             # {"ICD-10 Code": str, "ICD-10 Name": str}
RerankingResult = Dict[str, Any]          # {"Reranked ICD-10 Codes": List[RerankedCode]}

BatchMentionReranking = Dict[str, Any]    # {"mention_id": str, "Reranked ICD-10 Codes": List[RerankedCode]}
BatchRerankingResult = Dict[str, Any]     # {"Mention Rerankings": List[BatchMentionReranking]}


DiseaseInfo = Dict[str, Any]                 # {"Disease": str, "ICD10": str, "Supporting Evidence": List[str]}
DiseaseExtractionResult = Dict[str, Any]     # {"Diseases": List[DiseaseInfo]}
