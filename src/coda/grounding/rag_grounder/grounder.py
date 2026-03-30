"""
RAG-based grounder wrapper with ICD-10 defaults.
"""

from __future__ import annotations

import logging
from pathlib import Path

from gilda import Annotation, ScoredMatch
from gilda.process import normalize
from gilda.scorer import generate_match
from gilda.term import Term

from coda import CODA_BASE
from coda.grounding import BaseGrounder
from coda.llm_api import LLMClient, create_llm_client

from .config import RAGGrounderConfig
from .core import PipelineResult, RAGGrounderPipeline
from .ontology_adapters import load_icd10_retrieval_terms
from .retrieval_term import RetrievalTerm
from .setup import setup_retrieval_grounder

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = CODA_BASE.join(
    "rag_grounder",
    "indexes",
    name="term_index.icd10.pkl",
)


class RagGrounder(BaseGrounder):
    """
    RAG-based grounder with ICD-10 retrieval terms by default.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-5.4-mini",
        terms: list[RetrievalTerm] | None = None,
        cache_path: str | Path | None = None,
        force_rebuild: bool = False,
        retrieval_model_name: str = "all-MiniLM-L6-v2",
        concept_type: str = "disease",
        retrieval_top_k: int = 10,
        retrieval_min_similarity: float = 0.0,
    ):
        super().__init__()
        self._llm_client = llm_client
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self._terms = terms
        self._cache_path = Path(cache_path) if cache_path is not None else DEFAULT_CACHE_PATH
        self._force_rebuild = force_rebuild
        self._retrieval_model_name = retrieval_model_name
        self._concept_type = concept_type
        self._retrieval_top_k = retrieval_top_k
        self._retrieval_min_similarity = retrieval_min_similarity

        logger.info("Initializing RagGrounder with cache at %s", self._cache_path)
        terms = self._terms if self._terms is not None else load_icd10_retrieval_terms()
        term_store = setup_retrieval_grounder(
            terms=terms,
            model_name=self._retrieval_model_name,
            cache_path=self._cache_path,
            force_rebuild=self._force_rebuild,
        )
        llm_client = self._llm_client or create_llm_client(
            provider=self._llm_provider,
            model=self._llm_model,
        )
        config = RAGGrounderConfig(
            term_store=term_store,
            model_name=self._retrieval_model_name,
            concept_type=self._concept_type,
            retrieval_top_k=self._retrieval_top_k,
            retrieval_min_similarity=self._retrieval_min_similarity,
        )
        self._pipeline = RAGGrounderPipeline(llm_client=llm_client, config=config)

    def _get_pipeline(self) -> RAGGrounderPipeline:
        return self._pipeline

    @staticmethod
    def _make_term(term: RetrievalTerm) -> Term:
        if ":" in term.id:
            db, raw_id = term.id.split(":", 1)
        else:
            db, raw_id = "icd10", term.id
        entry_name = term.name.strip() if term.name else raw_id
        text = entry_name
        norm_text = normalize(text) if text else normalize(raw_id)
        return Term(
            norm_text=norm_text,
            text=text,
            db=db,
            id=raw_id,
            entry_name=entry_name,
            status="name",
            source="coda_rag_grounder",
        )

    def _annotation_matches(self, concept, span_text: str) -> list[ScoredMatch]:
        score_by_id = {
            term.id: float(score)
            for term, score in concept.retrieved_terms
        }

        candidate_terms = concept.reranked_terms
        if not candidate_terms:
            candidate_terms = [term for term, _ in concept.retrieved_terms]

        matches: list[ScoredMatch] = []
        for term in candidate_terms:
            term_obj = self._make_term(term)
            matches.append(
                ScoredMatch(
                    term=term_obj,
                    score=score_by_id.get(term.id, 0.0),
                    match=generate_match(span_text, term_obj.text),
                )
            )
        return matches

    def process(self, text: str) -> PipelineResult:
        """
        Process text through the full RAG pipeline.

        This is the canonical API for concept-level grounding.
        """
        if not text or not text.strip():
            return PipelineResult(text=text, Concepts=[])
        return self._get_pipeline().process(text)

    def ground(self, text: str) -> list[ScoredMatch]:
        """Compatibility shim returning top match per extracted concept."""
        result = self.process(text)
        matches: list[ScoredMatch] = []
        for concept in result.Concepts:
            span_text = concept.Concept or text
            concept_matches = self._annotation_matches(concept, span_text=span_text)
            if concept_matches:
                matches.append(concept_matches[0])
        return matches

    def annotate(self, text: str) -> list[Annotation]:
        """
        Compatibility shim returning mention-style annotations.

        Annotations are emitted for explicit concept mentions found in the
        input text, with concept-level matches attached.
        """
        result = self.process(text)
        annotations: list[Annotation] = []

        for concept in result.Concepts:
            concept_text = concept.Concept.strip()
            if not concept_text:
                continue

            matches = self._annotation_matches(concept, span_text=concept_text)
            if not matches:
                continue

            start = 0
            end = len(concept_text)
            annotations.append(
                Annotation(
                    text=concept_text,
                    matches=matches,
                    start=start,
                    end=end,
                )
            )
        return annotations
