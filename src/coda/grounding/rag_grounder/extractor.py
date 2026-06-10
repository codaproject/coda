"""
LLM-based concept extraction from text.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict

from coda.llm_api import LLMClient

from .config import PromptConfig

logger = logging.getLogger(__name__)


class Extractor:
    """Extract concepts and supporting evidence from text using LLM."""

    def __init__(
        self,
        concept_type: str,
        prompt_config_path: str,
        llm_client: LLMClient,
    ):
        self.llm_client = llm_client
        self.concept_type = concept_type
        self.config = PromptConfig.from_yaml(prompt_config_path)

    def extract(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"Concepts": []}

        concept_type_cap = self.concept_type.capitalize()
        system_prompt = self.config.system_prompt.format(concept_type=concept_type_cap)
        user_prompt = self.config.user_prompt.format(
            concept_type=concept_type_cap,
            text=text,
        )

        logger.debug(
            "--- Extractor Input ---\n[System Prompt]\n%s\n\n[User Prompt]\n%s",
            system_prompt, user_prompt,
        )

        try:
            if self.config.use_schema:
                response_json = self.llm_client.call_with_schema(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=self.config.schema,
                    schema_name=f"{self.concept_type}_extraction",
                    max_retries=3,
                    retry_delay=1.0,
                )
                logger.debug("--- Extractor Raw Output ---\n%s", json.dumps(response_json, indent=2))
                if response_json.get("api_failed", False):
                    logger.error("LLM API call failed")
                    return {"Concepts": []}
                concepts_raw = response_json.get("Concepts", [])
            else:
                response_text = self.llm_client.call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                try:
                    logger.debug("--- Extractor Raw Output ---\n%s", json.dumps(json.loads(response_text), indent=2))
                except (json.JSONDecodeError, TypeError):
                    logger.debug("--- Extractor Raw Output ---\n%s", response_text)
                try:
                    concepts_raw = json.loads(response_text)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to parse LLM response as JSON")
                    return {"Concepts": []}

            if not isinstance(concepts_raw, list):
                logger.warning("Unexpected response: concepts is not a list")
                return {"Concepts": []}

            concept_field = self.config.concept_key
            evidence_field = self.config.supporting_evidence_key
            concepts = []
            for c in concepts_raw:
                if not isinstance(c, dict):
                    continue
                concept = c.get(concept_field, "")
                evidence = c.get(evidence_field, [])
                if concept:
                    concepts.append({"Concept": concept, "Supporting_Evidence": evidence})
            return {"Concepts": concepts}

        except Exception as e:
            logger.error(f"Failed to extract concepts: {e}", exc_info=True)
            return {"Concepts": []}
