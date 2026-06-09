"""
RAG-based grounder.
"""
import logging
import time
from pathlib import Path

from gilda import Annotation, ScoredMatch
from gilda.process import normalize
from gilda.scorer import generate_match
from gilda.term import Term
from tqdm import tqdm

from coda.grounding import BaseGrounder
from coda.llm_api import LLMClient, create_llm_client

from .config import RAGGrounderConfig
from .extractor import Extractor
from .reranker import Reranker
from .retriever import Retriever
from .utils import find_evidence_spans
from .types import RetrievalTerm

logger = logging.getLogger(__name__)


class RagGrounder(BaseGrounder):
    """
    RAG-based grounder that extracts concepts from text and grounds them
    to ontology terms.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        llm_client: LLMClient | None = None,
    ):
        super().__init__()
        # Initialize config from yaml if provided, else use default yaml file
        if config_path is not None:
            self.config = RAGGrounderConfig.from_yaml(config_path)
        else:
            self.config = RAGGrounderConfig.default()

        # Initialize LLM client 
        if llm_client is None:
            self.llm_client = create_llm_client(provider="openai", model="gpt-4o-mini")
        else:
            self.llm_client = llm_client

        # Initialize Extractor from config and client
        self.extractor = Extractor(
            concept_type=self.config.concept_type,
            prompt_config_path=self.config.extractor.prompt_config_path,
            llm_client=self.llm_client
        )
        # Initialize Retriever from config and client
        self.retriever = Retriever(
            ontology=self.config.retriever.ontology,
            model_name=self.config.retriever.embedding_model,
            top_k=self.config.retriever.top_k,
            min_similarity=self.config.retriever.min_similarity,
        )
        # Initialize Reranker from config and client
        self.reranker = Reranker(
            llm_client=self.llm_client,
            prompt_config_path=self.config.reranker.prompt_config_path,
        )

    @staticmethod
    def _make_term(term: RetrievalTerm) -> Term:
        db, raw_id = term.id.split(":", 1)
        entry_name = term.name.strip() if term.name else raw_id
        norm_text = normalize(entry_name) if entry_name else normalize(raw_id)
        return Term(
            norm_text=norm_text,
            text=entry_name,
            db=db,
            id=raw_id,
            entry_name=entry_name,
            status="name",
            source="coda_rag_grounder",
        )

    def _annotation_matches(self, concept: dict, span_text: str) -> list[ScoredMatch]:
        return [
            ScoredMatch(
                term=self._make_term(term),
                score=score,
                match=generate_match(span_text, term.name),
            )
            for term, score in concept["matched_terms"]
        ]

    def process(self, text: str) -> dict:
        if not text or not text.strip():
            return {"text": text, "Concepts": []}

        logger.info("Starting RAG grounder pipeline")
        total_start = time.time()
        step_times = {}

        step1_start = time.time()
        extraction_result = self.extractor.extract(text)
        step_times["extraction"] = time.time() - step1_start

        concepts_raw = extraction_result.get("Concepts", [])
        logger.info(f"Extraction completed in {step_times['extraction']:.2f}s, found {len(concepts_raw)} concept(s)")

        if not concepts_raw:
            return {"text": text, "Concepts": []}

        concepts = [
            {
                "Concept": c.get("Concept", ""),
                "supporting_evidence": c.get("Supporting_Evidence", []),
                "matched_terms": [],
            }
            for c in concepts_raw
        ]

        step2_start = time.time()
        for concept in tqdm(concepts, desc="Retrieving terms", leave=False, disable=len(concepts) <= 1):
            concept_name = concept["Concept"]
            evidence_text = "\n".join(concept["supporting_evidence"])
            retrieval_text = f"{concept_name}\n\n{evidence_text}" if evidence_text else concept_name
            concept["matched_terms"] = self.retriever.retrieve(retrieval_text)
            logger.debug(f"Retrieved {len(concept['matched_terms'])} terms for: {concept_name}")
            
        step_times["retrieval"] = time.time() - step2_start
        logger.info(f"Retrieval completed in {step_times['retrieval']:.2f}s")

        step3_start = time.time()
        for concept in tqdm(concepts, desc="Reranking terms", leave=False, disable=len(concepts) <= 1):
            concept_name = concept["Concept"]
            score_by_id = {term.id: score for term, score in concept["matched_terms"]}
            reranked = self.reranker.rerank(
                concept=concept_name,
                evidences=concept["supporting_evidence"],
                retrieved_terms=[term for term, _ in concept["matched_terms"]],
            )
            concept["matched_terms"] = [(term, score_by_id.get(term.id, 0.0)) for term in reranked]
            logger.debug(f"Re-ranked {len(concept['matched_terms'])} terms for: {concept_name}")
        step_times["reranking"] = time.time() - step3_start
        logger.info(f"Re-ranking completed in {step_times['reranking']:.2f}s")

        total_time = time.time() - total_start
        logger.info(
            f"Pipeline timing: "
            f"Extraction={step_times['extraction']:.2f}s, "
            f"Retrieval={step_times['retrieval']:.2f}s, "
            f"Re-ranking={step_times['reranking']:.2f}s, "
            f"Total={total_time:.2f}s"
        )

        return {"text": text, "Concepts": concepts}

    def ground(self, text: str) -> list[ScoredMatch]:
        result = self.process(text)
        matches: list[ScoredMatch] = []
        for concept in result["Concepts"]:
            concept_matches = self._annotation_matches(concept, span_text=concept["Concept"] or text)
            if concept_matches:
                matches.append(concept_matches[0])
        return matches

    def annotate(self, text: str, min_similarity: float = 0.7) -> list[Annotation]:
        result = self.process(text)
        annotations: list[Annotation] = []
        for concept in result["Concepts"]:
            if not concept["Concept"].strip():
                continue
            matches = self._annotation_matches(concept, span_text=concept["Concept"])
            if not matches:
                continue
            spans = find_evidence_spans(text, concept["supporting_evidence"], min_similarity=min_similarity)
            for start, end, _match_type, _matched_text, _similarity in spans:
                if start < 0 or end > len(text) or end <= start:
                    continue
                annotations.append(
                    Annotation(
                        text=text[start:end],
                        matches=matches,
                        start=start,
                        end=end,
                    )
                )
        return annotations
