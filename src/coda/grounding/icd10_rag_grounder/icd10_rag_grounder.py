"""
RAG-based grounder for medical coding using ICD-10 codes.

This grounder uses a RAG (Retrieval-Augmented Generation) pipeline to extract
diseases from clinical text and assign ICD-10 codes.
"""

import logging
from typing import List

from .. import BaseGrounder
from .icd10_rag_extraction.pipeline import MedCoderPipeline
from coda.llm_api import create_llm_client

logger = logging.getLogger(__name__)

# Import gilda types - we'll use gilda's actual classes
from gilda import ScoredMatch, Annotation, Term
from gilda.scorer import Match


class RAGGrounder(BaseGrounder):
    """
    RAG-based grounder that uses MedCoderPipeline to extract diseases and assign ICD-10 codes.

    This grounder implements the BaseGrounder interface, converting ICD-10 codes
    to a format compatible with gilda's ScoredMatch and Annotation types.
    """

    def __init__(
        self,
        retrieval_top_k: int = 10,
        retrieval_min_similarity: float = 0.0,
        annotation_min_similarity: float = 0.5,
        **llm_kwargs
    ):
        """Initialize the RAG grounder.

        Parameters
        ----------
        retrieval_top_k : int
            Number of codes to retrieve per disease. Defaults to 10.
        retrieval_min_similarity : float
            Minimum similarity threshold for retrieval. Defaults to 0.0.
        annotation_min_similarity : float
            Minimum similarity threshold for evidence annotation. Defaults to 0.5.
        **llm_kwargs
            Arguments passed to create_llm_client().
            Common arguments:
            - model: str (e.g., "gpt-4o-mini", "llama3.2") - defaults to "gpt-4o-mini"
            - provider: str (e.g., "openai", "ollama") - auto-detected from model if not specified
            - api_key: str - API key for the provider
            See create_llm_client() documentation for full list.

        Notes
        -----
        Embeddings are automatically loaded from openacme's default location.
        Use openacme.generate_embeddings.generate_icd10_embeddings() to generate
        embeddings if they don't exist yet.

        Examples
        --------
        # Default (OpenAI, gpt-4o-mini)
        grounder = RAGGrounder()

        # Custom model
        grounder = RAGGrounder(model="llama3.2")

        # With API key
        grounder = RAGGrounder(model="gpt-4o-mini", api_key="sk-...")
        """
        llm_client = create_llm_client(**llm_kwargs)
        self.pipeline = MedCoderPipeline(
            llm_client=llm_client,
            retrieval_top_k=retrieval_top_k,
            retrieval_min_similarity=retrieval_min_similarity,
        )
        self.annotation_min_similarity = annotation_min_similarity

    def ground(self, text: str) -> List:
        """Ground text to ICD-10 codes.

        Parameters
        ----------
        text : str
            Clinical text to ground.

        Returns
        -------
        list
            List of gilda ScoredMatch objects.
        """
        logger.debug(f"Grounding text: {text[:100]}...")

        # Process through pipeline
        annotated_result = self.pipeline.process(
            text,
            text_type="clinical_note",
            identifier="",
            annotate_evidence=False,
            annotation_min_similarity=self.annotation_min_similarity
        )

        # process() always returns a dict, not a list
        if not isinstance(annotated_result, dict):
            logger.warning("Unexpected result type from pipeline.process()")
            return []

        # Extract all codes from the annotations
        scored_matches = []
        
        annotations = annotated_result.get('annotations', [])
        for annotation in annotations:
            # Get top match (first in ranked_matches)
            ranked_matches = annotation.get('ranked_matches', [])
            if not ranked_matches:
                continue

            top_match = ranked_matches[0]
            # Extract code from CURIE identifier (icd10:CODE -> CODE)
            identifier = top_match.get('identifier', '')
            code = identifier.replace("icd10:", "") if identifier.startswith("icd10:") else ""
            name = top_match.get('name', '')
            score = float(top_match.get('score', 0.0))

            # Create gilda Term object
            term = Term(
                norm_text=name.lower(),
                text=name,
                db="ICD10",
                id=code,
                entry_name=name,
                status="name",
                source="ICD10"
            )

            # Create gilda Match object (minimal - just query and ref)
            match = Match(query=name, ref=name)

            # Create gilda ScoredMatch
            scored_matches.append(
                ScoredMatch(term=term, score=max(0.0, min(1.0, score)), match=match)
            )

        logger.debug(f"Found {len(scored_matches)} scored matches")

        return scored_matches

    def annotate(self, text: str) -> List:
        """Annotate text with ICD-10 codes and evidence spans.

        Parameters
        ----------
        text : str
            Clinical text to annotate.

        Returns
        -------
        list
            List of gilda Annotation objects.
        """
        logger.debug(f"Annotating text: {text[:100]}...")

        # Process through pipeline with evidence annotation
        annotated_result = self.pipeline.process(
            text,
            text_type="clinical_note",
            identifier="",
            annotate_evidence=True,
            annotation_min_similarity=self.annotation_min_similarity
        )

        # process() always returns a dict, not a list
        if not isinstance(annotated_result, dict):
            logger.warning("Unexpected result type from pipeline.process()")
            return []

        annotations = []
        
        # Use annotations directly from the process() output
        pipeline_annotations = annotated_result.get('annotations', [])
        
        for annotation in pipeline_annotations:
            # Get the span text (already matched to original text)
            span_text = annotation.get('text', '')
            if not span_text:
                continue

            # Get top match
            ranked_matches = annotation.get('ranked_matches', [])
            if not ranked_matches:
                continue

            top_match_data = ranked_matches[0]
            identifier = top_match_data.get('identifier', '')
            
            # Extract code from CURIE identifier (icd10:CODE -> CODE)
            code = identifier.replace("icd10:", "") if identifier.startswith("icd10:") else ""
            name = top_match_data.get('name', '')
            score = float(top_match_data.get('score', 0.0))

            # Create gilda Term object
            term = Term(
                norm_text=name.lower(),
                text=name,
                db="ICD10",
                id=code,
                entry_name=name,
                status="name",
                source="ICD10"
            )

            # Create gilda Match object
            match = Match(query=name, ref=name)

            # Create gilda ScoredMatch
            scored_match = ScoredMatch(
                term=term, 
                score=max(0.0, min(1.0, score)), 
                match=match
            )

            # Get span positions from annotation properties (already computed by pipeline)
            annotation_properties = annotation.get('properties', {})
            start = annotation_properties.get('span_start')
            end = annotation_properties.get('span_end')
            
            # Fallback to text length if positions not available
            if start is None or end is None:
                logger.debug(f"Span positions not available for '{span_text[:60]}...', using fallback")
                start = 0
                end = len(span_text)
            
            annotations.append(
                Annotation(text=span_text, matches=[scored_match], start=start, end=end)
            )

        logger.debug(f"Created {len(annotations)} annotations")

        return annotations
