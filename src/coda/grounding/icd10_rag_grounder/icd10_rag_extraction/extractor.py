"""
LLM-based extraction from clinical notes.

Supports multiple extraction schemas for different use cases.
"""

import json
import os
import time
import logging
from typing import Dict, Any, Optional, Callable
from openai import OpenAI

from .schemas import DISEASE_EXTRACTION_SCHEMA, COD_EVIDENCE_EXTRACTION_SCHEMA
from .utils import validate_disease_extraction_result

logger = logging.getLogger(__name__)


class BaseExtractor:
    """
    Base class for LLM-based clinical text extractors.
    
    Handles common functionality: API client setup, retry logic, and LLM calls.
    Subclasses should implement validation and result processing.
    """

    def __init__(
        self,
        schema: Dict[str, Any],
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        schema_name: Optional[str] = None,
        default_empty_result: Optional[Dict[str, Any]] = None,
        validator: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        """
        Initialize the base extractor.

        Parameters
        ----------
        schema : Dict[str, Any]
            JSON schema for structured LLM output.
        api_key : str, optional
            OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
        model : str, default="gpt-4o-mini"
            OpenAI model to use.
        schema_name : str, optional
            Name for the schema (used in API call). If None, auto-detected from schema.
        default_empty_result : Dict[str, Any], optional
            Default result to return on empty input or validation failure. If None, auto-detected from schema.
        validator : Callable, optional
            Function to validate the LLM response structure. If None, uses basic validation.
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")

        self.client = OpenAI(api_key=api_key, timeout=(60.0, 300.0))
        self.model = model
        self.schema = schema
        
        # Auto-detect schema name and default empty result from schema
        if schema_name is None:
            schema_name = self._detect_schema_name(schema)
        self.schema_name = schema_name
        
        if default_empty_result is None:
            default_empty_result = self._detect_default_empty_result(schema)
        self.default_empty_result = default_empty_result
        
        if validator is None:
            validator = self._create_basic_validator(schema)
        self.validator = validator
        
        self.max_retries = 3
        self.retry_delay = 1.0

    def _detect_schema_name(self, schema: Dict[str, Any]) -> str:
        """Detect schema name from schema structure."""
        props = schema.get("properties", {})
        if "Diseases" in props:
            return "disease_extraction"
        elif "COD_EVIDENCE_SPANS" in props:
            return "cod_evidence_extraction"
        elif "Mentions" in props:
            return "mention_extraction"
        return "extraction"

    def _detect_default_empty_result(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Detect default empty result from schema structure."""
        props = schema.get("properties", {})
        if "Diseases" in props:
            return {"Diseases": []}
        elif "COD_EVIDENCE_SPANS" in props:
            return {"COD_EVIDENCE_SPANS": []}
        elif "Mentions" in props:
            return {"Mentions": []}
        return {}

    def _create_basic_validator(self, schema: Dict[str, Any]) -> Callable[[Dict[str, Any]], bool]:
        """Create a basic validator function from schema structure."""
        props = schema.get("properties", {})
        required = schema.get("required", [])
        
        def validator(result: Dict[str, Any]) -> bool:
            if not isinstance(result, dict):
                return False
            # Check required keys exist
            for key in required:
                if key not in result:
                    return False
                # Check type matches schema
                if key in props:
                    prop_type = props[key].get("type")
                    if prop_type == "array" and not isinstance(result[key], list):
                        return False
            return True
        
        return validator

    def extract(
        self,
        clinical_description: str,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract structured information from clinical text.

        Parameters
        ----------
        clinical_description : str
            Clinical text to extract from.
        system_prompt : str, optional
            System prompt for the LLM. If None, uses default.
        user_prompt_template : str, optional
            User prompt template. Should contain {clinical_description} placeholder.
            If None, uses default.

        Returns
        -------
        Dict[str, Any]
            Extracted structured data matching the schema.
        """
        if not clinical_description or not clinical_description.strip():
            return self.default_empty_result.copy()

        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        if user_prompt_template is None:
            user_prompt_template = self._get_default_user_prompt_template()

        user_prompt = user_prompt_template.format(clinical_description=clinical_description)

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
                            "name": self.schema_name,
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
                    result = self.default_empty_result.copy()
                    result["api_failed"] = True
                    return result

        if response_json is None:
            result = self.default_empty_result.copy()
            result["api_failed"] = True
            return result

        # Validate if validator provided
        if self.validator and not self.validator(response_json):
            logger.warning("Invalid response structure from LLM")
            return self.default_empty_result.copy()

        # Process and validate result
        return self._process_result(response_json, clinical_description)

    def _get_default_system_prompt(self) -> str:
        """Override in subclasses to provide default system prompt."""
        return "You are a medical coding assistant that extracts information from clinical descriptions.\n\n"

    def _get_default_user_prompt_template(self) -> str:
        """Override in subclasses to provide default user prompt template."""
        return "Extract information from the following clinical description:\n\n{clinical_description}"

    def _process_result(self, response_json: Dict[str, Any], clinical_description: str) -> Dict[str, Any]:
        """
        Process and validate the LLM response.
        
        Override in subclasses to implement schema-specific processing.
        """
        return response_json


class DiseaseExtractor(BaseExtractor):
    """
    Extract diseases/conditions with supporting evidence spans from clinical text.

    Returns disease-grouped format: {"Diseases": [{"Disease": str, "Supporting Evidence": [str, ...]}, ...]}
    The pipeline then flattens this into mentions for processing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        schema: Dict[str, Any] = DISEASE_EXTRACTION_SCHEMA,
    ):
        super().__init__(
            schema=schema,
            api_key=api_key,
            model=model,
            schema_name="disease_extraction",
            default_empty_result={"Diseases": []},
            validator=validate_disease_extraction_result,
        )

    def _get_default_system_prompt(self) -> str:
        return (
            "You are a medical coding assistant that extracts diseases and supporting evidence "
            "from clinical descriptions.\n\n"
        )

    def _get_default_user_prompt_template(self) -> str:
        return "Extract diseases and supporting evidence from the following clinical description:\n\n{clinical_description}"

    def _process_result(self, response_json: Dict[str, Any], clinical_description: str) -> Dict[str, Any]:
        """Process and validate disease extraction result."""
        clinical_lower = clinical_description.lower()
        validated_diseases = []

        for disease_item in response_json.get("Diseases", []):
            if not isinstance(disease_item, dict):
                continue

            disease_label = (disease_item.get("Disease") or "").strip()
            evidence_list = disease_item.get("Supporting Evidence", [])

            if not disease_label:
                continue

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


class CodEvidenceExtractor(BaseExtractor):
    """
    Extract clinical evidence spans as flat spans from clinical text (non-hierarchical extraction).

    Returns flat evidence format: {"COD_EVIDENCE_SPANS": [str, ...]}
    This is a general-purpose extractor that does not assume hierarchical disease grouping.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        schema: Dict[str, Any] = COD_EVIDENCE_EXTRACTION_SCHEMA,
    ):
        super().__init__(
            schema=schema,
            api_key=api_key,
            model=model,
            schema_name="cod_evidence_extraction",
            default_empty_result={"COD_EVIDENCE_SPANS": []},
        )

    def _get_default_system_prompt(self) -> str:
        return (
            "You are a medical coding assistant that extracts clinical evidence spans "
            "from clinical descriptions.\n\n"
        )

    def _get_default_user_prompt_template(self) -> str:
        return "Extract clinical evidence spans from the following clinical description:\n\n{clinical_description}"

    def _process_result(self, response_json: Dict[str, Any], clinical_description: str) -> Dict[str, Any]:
        """Process and validate COD evidence spans extraction result."""
        # Validate structure
        if not isinstance(response_json, dict) or "COD_EVIDENCE_SPANS" not in response_json:
            logger.warning("Invalid response structure from LLM - expected 'COD_EVIDENCE_SPANS' key")
            return {"COD_EVIDENCE_SPANS": []}
        if not isinstance(response_json["COD_EVIDENCE_SPANS"], list):
            logger.warning("Invalid response structure from LLM - 'COD_EVIDENCE_SPANS' must be a list")
            return {"COD_EVIDENCE_SPANS": []}

        clinical_lower = clinical_description.lower()
        validated_evidence = []

        for evidence_span in response_json.get("COD_EVIDENCE_SPANS", []):
            if not isinstance(evidence_span, str):
                continue

            evidence_span = evidence_span.strip()
            if not evidence_span:
                continue

            # Check if evidence span is verbatim (case-insensitive)
            if evidence_span.lower() not in clinical_lower:
                logger.warning(
                    f"Evidence span '{evidence_span[:60]}...' may not be verbatim from input text"
                )

            validated_evidence.append(evidence_span)

        return {"COD_EVIDENCE_SPANS": validated_evidence}
