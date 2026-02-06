"""
LLM-based re-ranking of retrieved terms.
"""

import logging
from typing import List, Optional

from pydantic import ValidationError
from coda.llm_api import LLMClient
from .schemas import (
    RERANKING_SCHEMA,
    BATCH_RERANKING_SCHEMA,
    RerankingResult,
    BatchRerankingResult
)
from ..retrieval_term import RetrievalTerm

logger = logging.getLogger(__name__)


class Reranker:
    """
    Re-rank retrieved terms using LLM reasoning.
    """

    def __init__(
        self,
        llm_client: LLMClient
    ):
        """Initialize reranker.
        
        Parameters
        ----------
        llm_client : LLMClient
            LLM client instance for making API calls.
        """
        self.llm_client = llm_client
        self.schema = RERANKING_SCHEMA
        self.batch_schema = BATCH_RERANKING_SCHEMA

    def rerank(
        self,
        concept: str,
        evidences: List[str],
        retrieved_terms: List[RetrievalTerm],
        system_prompt: Optional[str] = None
    ) -> List[RetrievalTerm]:
        """Re-rank retrieved terms based on concept and evidence.
        
        Parameters
        ----------
        concept : str
            Concept name.
        evidences : list of str
            List of supporting evidence strings.
        retrieved_terms : list of RetrievalTerm
            List of retrieved terms to rerank.
        system_prompt : str, optional
            Optional custom system prompt.
        
        Returns
        -------
        list of RetrievalTerm
            Reranked list of RetrievalTerm objects, ordered from most to least appropriate.
        """
        if not retrieved_terms:
            return []
        
        # Format retrieved terms for prompt
        retrieved_terms_formatted = []
        for term in retrieved_terms:
            term_info = f"  - Identifier: {term.id}, Name: {term.text}"
            if term.definition:
                term_info += f", Definition: {term.definition[:100]}"
            retrieved_terms_formatted.append(term_info)
        
        if system_prompt is None:
            system_prompt = """You are an expert that re-ranks retrieved terms based on how well they match a concept and its supporting evidence.

Consider these factors (in order of importance):
1. **Concept alignment**: Does the term accurately represent the concept?
2. **Evidence alignment**: Does the term match the supporting evidence?
3. **Specificity**: Prefer more specific terms over general ones when appropriate
4. **Retrieval relevance**: Consider how well the term matches semantically

Return ONLY JSON that matches the provided schema, ordered from most to least appropriate."""
        
        evidence_text = "\n".join(f"  - {e}" for e in evidences) if evidences else "  (No specific evidence provided)"
        
        user_prompt = f"""Given this concept and evidences, rerank these terms:

Concept:
{concept}

Supporting evidence:
{evidence_text}

Retrieved candidate terms:
{"\n".join(retrieved_terms_formatted)}

Re-rank these terms based on how well they match the concept and evidence."""
        
        try:
            # Use LLMClient's call_with_schema method
            response_json = self.llm_client.call_with_schema(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=self.schema,
                schema_name="reranking",
                max_retries=3,
                retry_delay=1.0
            )
            
            # Check for API failures
            if response_json.get("api_failed", False):
                logger.error("LLM API call failed")
                return []
            
            # Validate using Pydantic
            try:
                validated_result = RerankingResult.model_validate(response_json)
                # Convert back to dict
                result_dict = validated_result.model_dump()
            except ValidationError as e:
                logger.warning(f"Invalid response structure from LLM: {e}")
                return []
            
            # Create mapping from term ID to RetrievalTerm object
            term_id_to_term = {term.id: term for term in retrieved_terms}
            
            # Map reranked term IDs back to RetrievalTerm objects
            reranked_terms = []
            for term_info in result_dict['Reranked_Terms']:
                term_id = term_info.get('Term_Identifier', '')
                if term_id in term_id_to_term:
                    reranked_terms.append(term_id_to_term[term_id])
                else:
                    logger.warning(f"Term ID '{term_id}' from reranking not found in retrieved terms")
            
            return reranked_terms
            
        except Exception as e:
            logger.error(f"Failed to rerank terms: {e}", exc_info=True)
            return []

    def rerank_batch(
        self,
        concepts: List[str],
        evidences_list: List[List[str]],
        retrieved_terms_list: List[List[RetrievalTerm]],
        system_prompt: Optional[str] = None
    ) -> List[List[RetrievalTerm]]:
        """Re-rank retrieved terms for multiple concepts in a single API call.
        
        Parameters
        ----------
        concepts : list of str
            List of concept names.
        evidences_list : list of list of str
            List of evidence lists, one per concept.
        retrieved_terms_list : list of list of RetrievalTerm
            List of retrieved term lists, one per concept.
        system_prompt : str, optional
            Optional custom system prompt.
        
        Returns
        -------
        list of list of RetrievalTerm
            List of reranked term lists, one per concept, each ordered from most to least appropriate.
        """
        if not concepts or not retrieved_terms_list:
            return []
        
        if len(concepts) != len(evidences_list) or len(concepts) != len(retrieved_terms_list):
            logger.error("concepts, evidences_list, and retrieved_terms_list must have the same length")
            return []
        
        # Format all concepts and their retrieved terms for prompt
        concept_sections = []
        for i, (concept, evidences, retrieved_terms) in enumerate(zip(concepts, evidences_list, retrieved_terms_list)):
            evidence_text = "\n".join(f"  - {e}" for e in evidences) if evidences else "  (No specific evidence provided)"
            
            retrieved_terms_formatted = []
            for term in retrieved_terms:
                term_info = f"    - Identifier: {term.id}, Name: {term.text}"
                if term.definition:
                    term_info += f", Definition: {term.definition[:100]}"
                retrieved_terms_formatted.append(term_info)
            
            concept_section = f"""Concept {i + 1}:
  Concept: {concept}
  Supporting evidence:
{evidence_text}
  Retrieved candidate terms:
{"\n".join(retrieved_terms_formatted)}"""
            concept_sections.append(concept_section)
        
        if system_prompt is None:
            system_prompt = """You are an expert that re-ranks retrieved terms for multiple concepts based on how well they match each concept and its supporting evidence.

For each concept, consider these factors (in order of importance):
1. **Concept alignment**: Does the term accurately represent the concept?
2. **Evidence alignment**: Does the term match the supporting evidence?
3. **Specificity**: Prefer more specific terms over general ones when appropriate
4. **Retrieval relevance**: Consider how well the term matches semantically

Return ONLY JSON that matches the provided schema. The output should be an array of objects, where each object contains a Concept_Index (0-based) and the reranked terms for that concept, ordered from most to least appropriate."""
        
        user_prompt = f"""Given these concepts and their evidences, rerank the terms for each concept:

{"\n\n".join(concept_sections)}

Re-rank the terms for each concept based on how well they match the concept and evidence. Return one reranked list per concept, with each result object containing the Concept_Index (0-based) and its reranked terms."""
        
        try:
            # Use LLMClient's call_with_schema method
            response_json = self.llm_client.call_with_schema(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=self.batch_schema,
                schema_name="batch_reranking",
                max_retries=3,
                retry_delay=1.0
            )
            
            # Check for API failures
            if response_json.get("api_failed", False):
                logger.error("LLM API call failed")
                return []
            
            # Validate using Pydantic
            try:
                validated_result = BatchRerankingResult.model_validate(response_json)
                # Convert back to dict
                result_dict = validated_result.model_dump()
            except ValidationError as e:
                logger.warning(f"Invalid response structure from LLM: {e}")
                return []
            
            # Create mappings from term ID to RetrievalTerm object for each concept
            term_id_to_term_mappings = []
            for retrieved_terms in retrieved_terms_list:
                term_id_to_term_mappings.append({term.id: term for term in retrieved_terms})
            
            # Map reranked term IDs back to RetrievalTerm objects for each concept
            # Sort by Concept_Index to ensure correct ordering even if LLM returns out of order
            reranked_terms_batch = result_dict['Reranked_Terms_Batch']
            reranked_terms_batch_sorted = sorted(reranked_terms_batch, key=lambda x: x['Concept_Index'])
            
            if len(reranked_terms_batch_sorted) != len(concepts):
                logger.warning(f"Expected {len(concepts)} reranked lists, got {len(reranked_terms_batch_sorted)}")
            
            # Initialize result list with empty lists
            reranked_batch = [[] for _ in range(len(concepts))]
            
            # Add reranked terms to the correct concept index for each concept
            for concept_result in reranked_terms_batch_sorted:
                concept_index = concept_result.get('Concept_Index')
                reranked_terms_info = concept_result.get('Reranked_Terms', [])
                
                if concept_index < 0 or concept_index >= len(term_id_to_term_mappings):
                    logger.warning(f"Concept_Index {concept_index} is out of range (0-{len(concepts)-1})")
                    continue
                
                term_id_to_term = term_id_to_term_mappings[concept_index]
                reranked_terms = []
                
                # Map term IDs from LLM response back to RetrievalTerm objects.
                # Only include terms that exist in the retrieved terms (filter out any hallucinated/invalid IDs).
                for term_info in reranked_terms_info:
                    term_id = term_info.get('Term_Identifier', '')
                    if term_id in term_id_to_term:
                        reranked_terms.append(term_id_to_term[term_id])
                    else:
                        logger.warning(f"Term ID '{term_id}' from reranking not found in retrieved terms for concept {concept_index}")
                
                reranked_batch[concept_index] = reranked_terms
            
            return reranked_batch
            
        except Exception as e:
            logger.error(f"Failed to rerank terms in batch: {e}", exc_info=True)
            return []
