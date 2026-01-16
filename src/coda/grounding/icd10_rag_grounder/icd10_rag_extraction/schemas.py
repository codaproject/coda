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
                    "ICD10": {
                        "type": "string",
                        "description": "The ICD-10 code that corresponds to the disease."
                    }
                },
                "required": ["Disease", "Supporting Evidence", "ICD10"],
                "additionalProperties": False
            }
        }
    },
    "required": ["Diseases"],
    "additionalProperties": False
}

# -------------------------------------------------------------------
# Mention extraction (disease / condition centered)
# -------------------------------------------------------------------
MENTION_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "Mentions": {
            "type": "array",
            "description": (
                "Clinical conditions extracted from the text. "
                "Each item should be a verbatim span describing a disease, disorder, complication, "
                "injury, abnormal finding, or cause-of-death relevant condition."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "Mention": {
                        "type": "string",
                        "description": (
                            "An EXACT verbatim span copied from the input text that expresses a clinical condition. "
                            "Choose the most informative self-contained span, and include relevant modifiers if present. "
                            "Avoid overly short fragments when a longer span captures the same meaning."
                        ),
                    },
                    "ICD10": {
                        "type": "string",
                        "description": (
                            "Optional ICD-10 guess for this mention span. "
                            "May be an empty string if uncertain."
                        ),
                    },
                },
                "required": ["Mention", "ICD10"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["Mentions"],
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
                        "type": "string",
                        "description": "The mention_id from the input payload. Used to join results back.",
                    },
                    "Reranked ICD-10 Codes": {
                        "type": "array",
                        "description": (
                            "ICD-10 codes ordered from most to least appropriate for this mention."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "ICD-10 Code": {"type": "string"},
                                "ICD-10 Name": {"type": "string"},
                            },
                            "required": ["ICD-10 Code", "ICD-10 Name"],
                            "additionalProperties": False,
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
