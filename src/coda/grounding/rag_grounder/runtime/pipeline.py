"""
Main pipeline orchestrator for RAG grounder.

Combines LLM extraction, semantic retrieval, and re-ranking for any concept type.
"""

import logging
import time
from typing import List, Optional, Tuple
from tqdm import tqdm
from pydantic import BaseModel, ConfigDict

from coda.llm_api import LLMClient
from .extractor import Extractor
from .retriever import Retriever
from .reranker import Reranker
from .utils import find_evidence_spans
from ..retrieval_term import RetrievalTerm


logger = logging.getLogger(__name__)


# Pydantic models for pipeline output
class EvidenceSpan(BaseModel):
    """Pydantic model for evidence span."""
    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    similarity: float
    match_type: str


class ProcessedConcept(BaseModel):
    """Pydantic model for a processed concept with retrieval and reranking results."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    Concept: str
    evidence_spans: List[EvidenceSpan] = []
    retrieved_terms: List[Tuple[RetrievalTerm, float]] = []  # (term, similarity) tuples
    reranked_terms: List[RetrievalTerm] = []


class PipelineResult(BaseModel):
    """Pydantic model for pipeline output."""
    text: str
    Concepts: List[ProcessedConcept]


class RAGGrounderPipeline:
    """
    Complete pipeline for extracting concepts and grounding them to retrieval terms.
    
    Combines LLM extraction, semantic retrieval, and re-ranking.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        concept_type: str,
        kg_dir: Optional[str] = None,
        retrieval_top_k: int = 10,
        retrieval_min_similarity: float = 0.0,
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        """Initialize the RAG grounder pipeline.
        
        Parameters
        ----------
        llm_client : LLMClient
            LLM client instance for making API calls.
        concept_type : str
            Type of concept to extract (e.g., "disease", "vaccine", "medication").
        kg_dir : str, optional
            Directory containing KG TSV files. If None, uses pystow default location.
        retrieval_top_k : int
            Number of terms to retrieve per concept. Defaults to 10.
        retrieval_min_similarity : float
            Minimum similarity threshold for retrieval. Defaults to 0.0.
        model_name : str
            SentenceTransformer model name. Defaults to 'all-MiniLM-L6-v2'.
            Should match the model used during setup.
        """
        self.llm_client = llm_client
        self.concept_type = concept_type
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_min_similarity = retrieval_min_similarity
        
        # Initialize components
        self.extractor = Extractor(llm_client=llm_client, concept_type=concept_type)
        self.retriever = Retriever(kg_dir=kg_dir, model_name=model_name)
        self.reranker = Reranker(llm_client=llm_client)
    
    def process(
        self,
        text: str,
        annotation_min_similarity: float = 0.7,
        use_batch_reranking: bool = True
    ) -> PipelineResult:
        """Process text through full pipeline.
        
        Parameters
        ----------
        text : str
            Text to process.
        annotation_min_similarity : float
            Minimum similarity threshold for evidence annotation (0.0-1.0).
            Defaults to 0.7.
        use_batch_reranking : bool
            If True, use batch reranking (single API call for all concepts).
            If False, rerank each concept separately. Defaults to True.
        
        Returns
        -------
        PipelineResult
            Pydantic model with processed concepts including evidence spans, retrieved terms, and reranked terms.
        """
        if not text or not text.strip():
            return PipelineResult(text=text, Concepts=[])
        
        logger.info("Starting RAG grounder pipeline")
        
        step_times = {}
        total_start = time.time()
        
        # Step 1: Extract concepts using LLM
        logger.debug("Step 1: Extracting concepts and supporting evidence")
        
        step1_start = time.time()
        extraction_result = self.extractor.extract(text)
        step1_time = time.time() - step1_start
        step_times['extraction'] = step1_time
        
        concepts_raw = extraction_result.get('Concepts', [])
        
        logger.info(f"Extraction completed in {step1_time:.2f}s, found {len(concepts_raw)} concept(s)")
        
        if not concepts_raw:
            logger.warning("No concepts extracted from text")
            return PipelineResult(text=text, Concepts=[])
        
        # Post-process extraction: add evidence spans
        concepts = []
        for concept_raw in concepts_raw:
            evidence = concept_raw.get('Supporting_Evidence', [])
            
            # Get evidence spans: list of (start, end, match_type, matched_text, similarity) tuples
            spans_tuples = find_evidence_spans(
                text,
                evidence,
                min_similarity=annotation_min_similarity
            )
            
            # Convert tuples to EvidenceSpan objects
            evidence_spans_list = [
                EvidenceSpan(
                    text=matched_text,
                    start=start,
                    end=end,
                    similarity=similarity,
                    match_type=match_type
                )
                for start, end, match_type, matched_text, similarity in spans_tuples
            ]
            
            concepts.append({
                'Concept': concept_raw.get('Concept', ''),
                'evidence_spans': evidence_spans_list,
                'retrieved_terms': [],
                'reranked_terms': []
            })
        
        # Step 2: Retrieve terms using semantic search
        logger.debug(f"Step 2: Retrieving top-{self.retrieval_top_k} similar terms for each concept")
        
        step2_start = time.time()
        
        concepts_iter = tqdm(concepts, desc="Retrieving terms", leave=False, disable=len(concepts) <= 1)
        for concept in concepts_iter:
            concept_name = concept['Concept']
            evidence_spans = concept['evidence_spans']
            
            # Extract matched text from evidence spans for retrieval
            evidence_texts = [span.text for span in evidence_spans]
            evidence_text = "\n".join(evidence_texts) if evidence_texts else ""
            retrieval_text = f"{concept_name}\n\n{evidence_text}" if evidence_text else concept_name
            
            # Retrieve terms (returns list of tuples: (RetrievalTerm, similarity))
            retrieved_terms = self.retriever.retrieve(
                retrieval_text,
                top_k=self.retrieval_top_k,
                min_similarity=self.retrieval_min_similarity
            )
            
            # Store tuples directly (term, similarity)
            concept['retrieved_terms'] = retrieved_terms
            concepts_iter.set_postfix({"concept": concept_name[:30]})
            logger.debug(f"Retrieved {len(retrieved_terms)} terms for concept: {concept_name}")
        
        step2_time = time.time() - step2_start
        step_times['retrieval'] = step2_time
        
        logger.info(f"Retrieval completed in {step2_time:.2f}s for {len(concepts)} concept(s)")
        
        # Step 3: Re-rank terms using LLM
        logger.debug("Step 3: Re-ranking terms")
        
        step3_start = time.time()
        
        if use_batch_reranking and len(concepts) > 1:
            # Use batch reranking for efficiency
            concepts_list = [c['Concept'] for c in concepts]
            # Extract matched text from evidence spans
            evidences_list = [[span.text for span in c['evidence_spans']] for c in concepts]
            # Extract just terms from tuples for reranking
            retrieved_terms_list = [[term for term, _ in c['retrieved_terms']] for c in concepts]
            
            reranked_batch = self.reranker.rerank_batch(
                concepts=concepts_list,
                evidences_list=evidences_list,
                retrieved_terms_list=retrieved_terms_list
            )
            
            # Assign reranked terms back to concepts
            for i, concept in enumerate(concepts):
                if i < len(reranked_batch):
                    concept['reranked_terms'] = reranked_batch[i]
                else:
                    concept['reranked_terms'] = []
                    logger.warning(f"No reranked terms returned for concept {i}")
        else:
            # Rerank each concept separately
            rerank_iter = tqdm(concepts, desc="Reranking terms", leave=False, disable=len(concepts) <= 1)
            for concept in rerank_iter:
                concept_name = concept['Concept']
                evidence_spans = concept['evidence_spans']
                # Extract just terms from tuples for reranking
                retrieved_terms = [term for term, _ in concept['retrieved_terms']]
                
                # Extract matched text from evidence spans
                evidence_texts = [span.text for span in evidence_spans]
                
                # Re-rank
                reranked_terms = self.reranker.rerank(
                    concept=concept_name,
                    evidences=evidence_texts,
                    retrieved_terms=retrieved_terms
                )
                
                concept['reranked_terms'] = reranked_terms
                rerank_iter.set_postfix({"concept": concept_name[:30]})
                logger.debug(f"Re-ranked {len(reranked_terms)} terms for concept: {concept_name}")
        
        step3_time = time.time() - step3_start
        step_times['reranking'] = step3_time
        
        logger.info(f"Re-ranking completed in {step3_time:.2f}s for {len(concepts)} concept(s)")
        
        total_time = time.time() - total_start
        step_times['total'] = total_time
        
        # Log timing breakdown
        logger.info(
            f"Pipeline timing breakdown: "
            f"Extraction={step_times['extraction']:.2f}s, "
            f"Retrieval={step_times['retrieval']:.2f}s, "
            f"Re-ranking={step_times['reranking']:.2f}s, "
            f"Total={step_times['total']:.2f}s"
        )
        
        logger.info("Pipeline completed")
        
        # Construct and return Pydantic model
        processed_concepts = []
        for concept in concepts:
            processed_concepts.append(ProcessedConcept(
                Concept=concept['Concept'],
                evidence_spans=concept['evidence_spans'],
                retrieved_terms=concept['retrieved_terms'],
                reranked_terms=concept['reranked_terms']
            ))
        return PipelineResult(text=text, Concepts=processed_concepts)
