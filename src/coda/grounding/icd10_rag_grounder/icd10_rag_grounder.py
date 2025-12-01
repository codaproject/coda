"""
RAG-based grounder for medical coding using ICD-10 codes.

This grounder uses a RAG (Retrieval-Augmented Generation) pipeline to extract
diseases from clinical text and assign ICD-10 codes.
"""

import logging
from typing import List, Optional, Dict, Any

from .. import BaseGrounder
from .icd10_rag_extraction.pipeline import MedCoderPipeline
from .icd10_rag_extraction.utils import get_icd10_name

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
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        retrieval_top_k: int = 10,
        retrieval_min_similarity: float = 0.0,
        annotation_min_similarity: float = 0.5
    ):
        """Initialize the RAG grounder.

        Parameters
        ----------
        openai_api_key : str, optional
            OpenAI API key. Defaults to OPENAI_API_KEY environment variable.
        openai_model : str
            OpenAI model name. Defaults to "gpt-4o-mini".
        retrieval_top_k : int
            Number of codes to retrieve per disease. Defaults to 10.
        retrieval_min_similarity : float
            Minimum similarity threshold for retrieval. Defaults to 0.0.
        annotation_min_similarity : float
            Minimum similarity threshold for evidence annotation. Defaults to 0.5.

        Notes
        -----
        Embeddings are automatically loaded from openacme's default location.
        Use openacme.generate_embeddings.generate_icd10_embeddings() to generate
        embeddings if they don't exist yet.
        """
        self.pipeline = MedCoderPipeline(
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            retrieval_top_k=retrieval_top_k,
            retrieval_min_similarity=retrieval_min_similarity
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
        result = self.pipeline.process(
            text,
            annotate_evidence=False,
            annotation_min_similarity=self.annotation_min_similarity
        )

        # Extract all codes from the result
        scored_matches = []
        diseases = result.get('Diseases', [])

        for disease in diseases:
            # Get reranked codes - take only the top (first) code
            codes = disease.get('reranked_codes', [])
            if not codes:
                continue

            # Get the top code (first in reranked list)
            code_info = codes[0]
            code = code_info.get('ICD-10 Code', '') or code_info.get('code', '')
            name = code_info.get('ICD-10 Name', '') or code_info.get('name', get_icd10_name(code))
            # Use similarity score from retrieval
            similarity = code_info.get('similarity', 0.0)
            score = float(similarity)

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
        result = self.pipeline.process(
            text,
            annotate_evidence=True,
            annotation_min_similarity=self.annotation_min_similarity
        )

        annotations = []
        diseases = result.get('Diseases', [])

        for disease in diseases:
            # Get evidence spans (primary source for annotation text)
            evidence_spans = disease.get('evidence_spans', [])
            if not evidence_spans:
                continue

            # Get reranked codes - take only the top (first) code
            codes = disease.get('reranked_codes', [])
            if not codes:
                codes = disease.get('retrieved_codes', [])
            if not codes:
                continue

            # Get the top code (first in reranked list)
            code_info = codes[0]
            code = code_info.get('ICD-10 Code', '')
            name = code_info.get('ICD-10 Name', '')
            # Use similarity score from retrieval
            similarity = code_info.get('similarity', 0.0)
            score = float(similarity)

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

            # Create single ScoredMatch for the top code
            top_match = ScoredMatch(term=term, score=max(0.0, min(1.0, score)), match=match)

            # Create one annotation per evidence span, each with the top code
            for span in evidence_spans:
                span_text = span.get('text', '')
                start = span.get('start', 0)
                end = span.get('end', len(span_text))
                if span_text:
                    annotations.append(
                        Annotation(text=span_text, matches=[top_match], start=start, end=end)
                    )

        logger.debug(f"Created {len(annotations)} annotations")

        return annotations
