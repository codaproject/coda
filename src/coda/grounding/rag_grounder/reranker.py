"""
LLM-based re-ranking of retrieved terms.
"""
import json
import logging
from typing import List

from coda.llm_api import LLMClient

from .config import PromptConfig
from .types import RetrievalTerm

logger = logging.getLogger(__name__)


class Reranker:
    """Re-rank retrieved terms using LLM reasoning."""

    def __init__(
        self, llm_client: LLMClient, 
        prompt_config_path: str
    ):
        self.llm_client = llm_client
        self.config = PromptConfig.from_yaml(prompt_config_path)

    def rerank(
        self,
        concept: str,
        evidences: List[str],
        retrieved_terms: List[RetrievalTerm],
    ) -> List[RetrievalTerm]:
        if not retrieved_terms:
            return []

        retrieved_terms_formatted = "\n".join(
            f"  - Identifier: {term.id}, Name: {term.name}"
            + (f", Definition: {term.definition}" if term.definition else "")
            for term in retrieved_terms
        )
        evidence_text = (
            "\n".join(f"  - {e}" for e in evidences)
            if evidences
            else "  (No specific evidence provided)"
        )
        user_prompt = self.config.user_prompt.format(
            concept=concept,
            evidence_text=evidence_text,
            retrieved_terms=retrieved_terms_formatted,
        )

        logger.debug(
            "--- Reranker Input ---\n[System Prompt]\n%s\n\n[User Prompt]\n%s",
            self.config.system_prompt, user_prompt,
        )

        try:
            response_json = self.llm_client.call_with_schema(
                system_prompt=self.config.system_prompt,
                user_prompt=user_prompt,
                schema=self.config.schema,
                schema_name="reranking",
                max_retries=3,
                retry_delay=1.0,
            )
            logger.debug("--- Reranker Raw Output ---\n%s", json.dumps(response_json, indent=2))

            if response_json.get("api_failed", False):
                logger.error("LLM API call failed")
                return []

            reranked = response_json.get("Reranked_Terms", [])
            if not isinstance(reranked, list):
                logger.warning("Unexpected response: 'Reranked_Terms' is not a list")
                return []

            term_id_to_term = {term.id: term for term in retrieved_terms}
            result = []
            for item in reranked:
                if not isinstance(item, dict):
                    continue
                term_id = item.get("Term_Identifier", "")
                if term_id in term_id_to_term:
                    result.append(term_id_to_term[term_id])
                else:
                    logger.warning(f"Term ID '{term_id}' not found in retrieved terms")
            return result

        except Exception as e:
            logger.error(f"Failed to rerank terms: {e}", exc_info=True)
            return []
