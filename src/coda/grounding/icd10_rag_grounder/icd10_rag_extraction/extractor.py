"""
LLM-based disease extraction from clinical notes.

Extracts diseases/conditions with supporting evidence spans, which are then
flattened into mentions for ICD-10 annotation.
"""

import json
import os
import time
import logging
from typing import Dict, Any, Optional
from openai import OpenAI

from .schemas import DISEASE_EXTRACTION_SCHEMA
from .utils import validate_disease_extraction_result, validate_icd10_code

logger = logging.getLogger(__name__)


class DiseaseExtractor:
    """
    Extract diseases/conditions with supporting evidence spans from clinical text.

    Returns disease-grouped format: {"Diseases": [{"Disease": str, "ICD10": str, "Supporting Evidence": [str, ...]}, ...]}
    The pipeline then flattens this into mentions for processing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini"
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")

        self.client = OpenAI(api_key=api_key, timeout=(60.0, 300.0))
        self.model = model
        self.schema = DISEASE_EXTRACTION_SCHEMA
        self.max_retries = 3
        self.retry_delay = 1.0

    def extract(
        self,
        clinical_description: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        if not clinical_description or not clinical_description.strip():
            return {"Diseases": []}

        if system_prompt is None:
            system_prompt = (
                "You are a medical coding assistant that extracts diseases and supporting evidence "
                "from clinical descriptions.\n\n"
                "CRITICAL: For 'Supporting Evidence', you MUST extract EXACT verbatim text spans "
                "from the input text. Do NOT paraphrase, reword, or summarize. Copy the text exactly "
                "as it appears in the clinical description.\n\n"
                "Example:\n"
                "Input: 'Patient has chest pain and shortness of breath.'\n"
                "Correct evidence: ['chest pain', 'shortness of breath']\n"
                "WRONG evidence: ['Patient presents with chest discomfort', 'difficulty breathing']\n\n"
                "Provide accurate ICD-10 codes for each identified disease."
            )

        user_prompt = (
                f"Extract diseases and supporting evidence from the following clinical description.\n\n"
                f"IMPORTANT: For 'Supporting Evidence', copy EXACT text spans from the description below. "
                f"Do not paraphrase or reword.\n\n"
                f"Clinical Description:\n{clinical_description}"
        )

        response_json = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "disease_extraction",
                            "schema": self.schema,
                            "strict": True,
                        }
                    },
                )
                response_json = json.loads(response.output_text)
                break
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Extraction attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} extraction attempts failed. Last error: {e}")
                    return {"Diseases": [], "api_failed": True}

        if response_json is None:
            return {"Diseases": [], "api_failed": True}

        if not validate_disease_extraction_result(response_json):
            logger.warning("Invalid response structure from LLM")
            return {"Diseases": []}

        clinical_lower = clinical_description.lower()
        validated_diseases = []

        for disease_item in response_json.get("Diseases", []):
            if not isinstance(disease_item, dict):
                continue

            disease_label = (disease_item.get("Disease") or "").strip()
            disease_icd10 = (disease_item.get("ICD10") or "").strip()
            evidence_list = disease_item.get("Supporting Evidence", [])

            if not disease_label:
                continue

            # Normalize ICD10 casing
            if isinstance(disease_icd10, str):
                disease_icd10 = disease_icd10.strip().upper()
                disease_item["ICD10"] = disease_icd10

            # Validate ICD10 existence if provided; clear if invalid
            if disease_icd10 and not validate_icd10_code(disease_icd10, check_existence=True):
                logger.warning(
                    f"Invalid/non-existent ICD-10 code '{disease_icd10}' for disease '{disease_label[:60]}...' — clearing code"
                )
                disease_item["ICD10"] = ""

            # Validate and clean evidence spans
            if not isinstance(evidence_list, list):
                evidence_list = []

            validated_evidence = []
            for ev in evidence_list:
                if not isinstance(ev, str):
                    continue
                ev_clean = ev.strip()
                if not ev_clean:
                    continue

                # Check if evidence is verbatim (case-insensitive)
                if ev_clean.lower() not in clinical_lower:
                    logger.warning(
                        f"Evidence span '{ev_clean[:60]}...' may not be verbatim from input text"
                    )

                validated_evidence.append(ev_clean)

            # Only include disease if it has at least one evidence span
            if validated_evidence:
                disease_item["Disease"] = disease_label
                disease_item["Supporting Evidence"] = validated_evidence
                validated_diseases.append(disease_item)

        return {"Diseases": validated_diseases}
