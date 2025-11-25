"""
RAG-based grounder for medical coding using ICD-10 codes.

This grounder uses a RAG (Retrieval-Augmented Generation) pipeline to extract
diseases from clinical text and assign ICD-10 codes.
"""

import logging
from typing import List, Optional, Dict, Any

from .. import BaseGrounder
from .icd10_map.pipeline import MedCoderPipeline
from .icd10_map.utils import get_icd10_name

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
        embeddings_dir: str = 'data/icd10_embeddings',
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        retrieval_top_k: int = 10,
        retrieval_min_similarity: float = 0.0,
        annotation_min_similarity: float = 0.5
    ):
        """
        Initialize the RAG grounder.
        
        Args:
            embeddings_dir: Directory containing ICD-10 embeddings
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            openai_model: OpenAI model name
            retrieval_top_k: Number of codes to retrieve per disease
            retrieval_min_similarity: Minimum similarity threshold for retrieval
            annotation_min_similarity: Minimum similarity threshold for evidence annotation
        """
        self.pipeline = MedCoderPipeline(
            embeddings_dir=embeddings_dir,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            retrieval_top_k=retrieval_top_k,
            retrieval_min_similarity=retrieval_min_similarity
        )
        self.annotation_min_similarity = annotation_min_similarity
        logger.info(f"RAGGrounder initialized with embeddings_dir={embeddings_dir}")
    
    def ground(self, text: str) -> List:
        """
        Ground text to ICD-10 codes.
        
        Args:
            text: Clinical text to ground
            
        Returns:
            List of gilda ScoredMatch objects
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
            # Get reranked codes
            codes = disease.get('reranked_codes', [])
            
            # Add reranked codes with decreasing scores
            for idx, code_info in enumerate(codes[:5]):  # Top 5 codes
                code = code_info.get('ICD-10 Code', '') or code_info.get('code', '')
                name = code_info.get('ICD-10 Name', '') or code_info.get('name', get_icd10_name(code))
                # Score decreases with rank (top code gets 0.9, then 0.8, 0.7, etc.)
                score = 0.9 - (idx * 0.1)
                
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
                    ScoredMatch(term=term, score=max(0.1, score), match=match)
                )
        
        logger.debug(f"Found {len(scored_matches)} scored matches")
        
        return scored_matches
    
    def annotate(self, text: str) -> List:
        """
        Annotate text with ICD-10 codes and evidence spans.
        
        Args:
            text: Clinical text to annotate
            
        Returns:
            List of gilda Annotation objects
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
            
            # Get reranked codes (preferred) or retrieved codes
            codes = disease.get('reranked_codes', [])
            if not codes:
                codes = disease.get('retrieved_codes', [])
            
            # Create matches for each code
            matches = []
            for idx, code_info in enumerate(codes[:5]):  # Top 5 codes
                code = code_info.get('ICD-10 Code', '') or code_info.get('code', '')
                name = code_info.get('ICD-10 Name', '') or code_info.get('name', get_icd10_name(code))
                # Score decreases with rank (top code gets 0.9, then 0.8, 0.7, etc.)
                score = 0.9 - (idx * 0.1)
                
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
                matches.append(
                    ScoredMatch(term=term, score=max(0.1, score), match=match)
                )
            
            # Create one annotation per evidence span
            if evidence_spans:
                for span in evidence_spans:
                    span_text = span.get('text', '')
                    start = span.get('start', 0)
                    end = span.get('end', len(span_text))
                    if span_text:
                        annotations.append(
                            Annotation(text=span_text, matches=matches, start=start, end=end)
                        )
        
        logger.debug(f"Created {len(annotations)} annotations")
        
        return annotations

