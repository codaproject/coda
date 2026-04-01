"""
LLM-based concept extraction from text.

Extracts concepts (e.g., diseases, vaccines) and supporting evidence
from input text using LLM with structured output.
"""
import logging
from typing import Any, Dict

from pydantic import ValidationError

from coda.llm_api import LLMClient

from .schemas import ExtractionResult, get_extraction_schema

logger = logging.getLogger(__name__)


class Extractor:
    """
    Extract concepts and supporting evidence from text using LLM.

    Similar to DiseaseExtractor but works with any concept type.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        concept_type: str,
    ):
        """
        Initialize concept extractor.

        Parameters
        ----------
        llm_client : LLMClient
            LLM client instance for making API calls.
        concept_type : str
            Type of concept to extract (e.g., "disease", "vaccine", "medication").
        """
        self.llm_client = llm_client
        self.concept_type = concept_type
        self.schema = get_extraction_schema(concept_type)

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract concepts and supporting evidence from text.

        Parameters
        ----------
        text : str
            Input text to extract concepts from.

        Returns
        -------
        Dict[str, Any]
            Dictionary with 'Concepts' list containing concept info.
            Each concept has: 'Concept', 'Supporting_Evidence'
        """
        if not text or not text.strip():
            return {"Concepts": []}

        concept_type_capitalized = self.concept_type.capitalize()

        system_prompt = (
            f"You are an assistant that extracts {concept_type_capitalized} concepts and supporting evidence "
            "from text.\n\n"
            "CRITICAL: For 'Supporting_Evidence', you MUST extract EXACT verbatim text spans "
            "from the input text. Do NOT paraphrase, reword, or summarize. Copy the text exactly "
            "as it appears in the input.\n\n"
            f"Extract all {concept_type_capitalized} concepts mentioned in the text and provide the supporting evidence for each concept."
        )

        try:
            user_prompt = (
                f"Extract {concept_type_capitalized} concepts and supporting evidence from the following text.\n\n"
                "IMPORTANT: For 'Supporting_Evidence', copy EXACT text spans from the text below. "
                "Do not paraphrase or reword.\n\n"
                f"Text:\n{text}"
            )

            response_json = self.llm_client.call_with_schema(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=self.schema,
                schema_name=f"{self.concept_type}_extraction",
                max_retries=3,
                retry_delay=1.0,
            )

            if response_json.get("api_failed", False):
                logger.error("LLM API call failed")
                return {"Concepts": []}

            try:
                validated_result = ExtractionResult.model_validate(response_json)
                return validated_result.model_dump()
            except ValidationError as e:
                logger.warning(f"Invalid response structure from LLM: {e}")
                return {"Concepts": []}

        except Exception as e:
            logger.error(f"Failed to extract concepts: {e}", exc_info=True)
            return {"Concepts": []}
