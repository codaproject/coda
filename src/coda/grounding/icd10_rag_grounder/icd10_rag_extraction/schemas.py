"""
JSON schemas for structured LLM outputs.
"""

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
                        "description": "The ICD-10 code."
                    },
                    "ICD-10 Name": {
                        "type": "string",
                        "description": "The human-readable name corresponding to the ICD-10 code."
                    }
                },
                "required": ["ICD-10 Code", "ICD-10 Name"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["Reranked ICD-10 Codes"],
    "additionalProperties": False,
}

# Type hints for better IDE support
# Note: TypedDict doesn't support keys with spaces, so we use Dict[str, Any] instead
from typing import Dict, Any, List

# Type aliases for documentation (actual validation uses JSON schemas above)
DiseaseInfo = Dict[str, Any]  # {"Disease": str, "Supporting Evidence": List[str], "ICD10": str}
DiseaseExtractionResult = Dict[str, Any]  # {"Diseases": List[DiseaseInfo]}
RerankedCode = Dict[str, Any]  # {"ICD-10 Code": str, "ICD-10 Name": str}
RerankingResult = Dict[str, Any]  # {"Reranked ICD-10 Codes": List[RerankedCode]}

