"""
LLM-based extraction from clinical notes.

Supports multiple extraction schemas for different use cases.
"""

import logging
from typing import Dict, Any, Optional, Callable

from coda.llm_api import LLMClient


logger = logging.getLogger(__name__)


# Configuration dictionaries for different extractor types
EXTRACTOR_CONFIGS = {
    "disease_extraction": {
        "system_prompt": (
            "You are a medical coding assistant that extracts diseases and supporting evidence "
            "from clinical descriptions.\n\n"
        ),
        "user_prompt_template": (
            "Extract diseases and supporting evidence from the following clinical description:\n\n"
            "{clinical_description}"
        ),
    },
    "cod_evidence_extraction": {
        "system_prompt": (
            "You are a medical coding assistant that extracts clinical evidence spans "
            "from clinical descriptions.\n\n"
        ),
        "user_prompt_template": (
            "Extract clinical evidence spans from the following clinical description:\n\n"
            "{clinical_description}"
        ),
    },
}


class BaseExtractor:
    """
    LLM-based clinical text extractor.
    
    Handles extraction with configurable prompts and schemas.
    Use EXTRACTOR_CONFIGS for common configurations, or provide custom prompts.
    """

    def __init__(
        self,
        schema: Dict[str, Any],
        llm_client: LLMClient,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
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
        llm_client : LLMClient
            LLM client adapter to use. Must be provided.
        system_prompt : str, optional
            System prompt for the LLM. If None, uses a minimal default.
        user_prompt_template : str, optional
            User prompt template. Should contain {clinical_description} placeholder.
            If None, uses a minimal default.
        schema_name : str, optional
            Name for the schema (used in API call). If None, auto-detected from schema.
        default_empty_result : Dict[str, Any], optional
            Default result to return on empty input or validation failure. If None, auto-detected from schema.
        validator : Callable, optional
            Function to validate the LLM response structure. If None, uses basic validation.
        """
        self.llm_client = llm_client
        self.schema = schema
        
        # Auto-detect schema name and default empty result from schema
        if schema_name is None:
            schema_name = self._detect_schema_name(schema)
        self.schema_name = schema_name
        
        # Auto-configure prompts from EXTRACTOR_CONFIGS if not provided
        if system_prompt is None or user_prompt_template is None:
            config = EXTRACTOR_CONFIGS.get(schema_name, {})
            if system_prompt is None:
                system_prompt = config.get("system_prompt", "You are a medical coding assistant.\n\n")
            if user_prompt_template is None:
                user_prompt_template = config.get("user_prompt_template", "{clinical_description}")
        
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        
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
            system_prompt = self.system_prompt

        if user_prompt_template is None:
            user_prompt_template = self.user_prompt_template

        user_prompt = user_prompt_template.format(clinical_description=clinical_description)

        # Use LLM adapter for API call
        response_json = self.llm_client.call_with_schema(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=self.schema,
            schema_name=self.schema_name,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )

        # Check for API failure
        if response_json.get("api_failed", False):
            result = self.default_empty_result.copy()
            result["api_failed"] = True
            return result

        # Validate if validator provided
        if self.validator and not self.validator(response_json):
            logger.warning(
                f"Invalid response structure from LLM. "
                f"Expected schema keys: {list(self.schema.get('properties', {}).keys())}, "
                f"Got keys: {list(response_json.keys()) if isinstance(response_json, dict) else 'not a dict'}"
            )
            logger.debug(f"Full response_json (first 500 chars): {str(response_json)[:500]}")
            return self.default_empty_result.copy()

        # Process and validate result
        return self._process_result(response_json, clinical_description)

    def _process_result(self, response_json: Dict[str, Any], clinical_description: str) -> Dict[str, Any]:
        """
        Process and validate the LLM response.
        
        Validates spans are verbatim for COD_EVIDENCE_SPANS and DISEASE schemas.
        """
        schema_name = self._detect_schema_name(self.schema)
        clinical_lower = clinical_description.lower()
        
        # For COD_EVIDENCE_SPANS schema, validate spans are verbatim
        if schema_name == "cod_evidence_extraction":
            if not isinstance(response_json, dict) or "COD_EVIDENCE_SPANS" not in response_json:
                return response_json
            if not isinstance(response_json["COD_EVIDENCE_SPANS"], list):
                return response_json
            
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
        
        # For DISEASE schema, validate evidence spans are verbatim
        elif schema_name == "disease_extraction":
            if not isinstance(response_json, dict) or "Diseases" not in response_json:
                return response_json
            
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
        
        return response_json
