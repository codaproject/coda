"""
RAG-based grounder.
"""
import logging
import time

from gilda import Annotation, ScoredMatch
from gilda.process import normalize
from gilda.scorer import generate_match
from gilda.term import Term
from tqdm import tqdm

from coda.config import PROMPTS, settings
from coda.grounding import BaseGrounder
from coda.llm_api import LLMClient, create_llm_client

from .extractor import Extractor, Hunflair2Extractor, LLMExtractor
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
        llm_client: LLMClient | None = None,
    ):
        super().__init__()
        # Snapshot the RAG config from the global settings into mutable instance
        # attributes (so update_config can change them at runtime without
        # mutating the shared global settings singleton).
        rag = settings.grounder.rag
        self.concept_type = rag.concept_type
        self.provider = rag.llm.provider
        self.model = rag.llm.model
        self.extractor_type = rag.extractor.type
        self.extractor_prompt = rag.extractor.prompt
        self.ontology = rag.retriever.ontology
        self.embedding_model = rag.retriever.embedding_model
        self.top_k = rag.retriever.top_k
        self.min_similarity = rag.retriever.min_similarity
        self.reranker_enabled = rag.reranker.enabled
        self.reranker_prompt = rag.reranker.prompt

        # Initialize LLM client
        if llm_client is None:
            self.llm_client = create_llm_client(
                provider=self.provider, model=self.model)
        else:
            self.llm_client = llm_client

        # Build pipeline components from config
        self.extractor = self._build_extractor()
        self.retriever = self._build_retriever()
        self.reranker = self._build_reranker()

    @staticmethod
    def _prompt(key: str) -> dict:
        """Look up a prompt config by key (a config/prompts/ filename stem)."""
        return PROMPTS[key]

    def _build_extractor(self) -> Extractor:
        match self.extractor_type.lower().strip():
            case "llm":
                return LLMExtractor(
                    concept_type=self.concept_type,
                    prompt_config=self._prompt(self.extractor_prompt),
                    llm_client=self.llm_client,
                )
            case "hunflair":
                return Hunflair2Extractor(concept_type=self.concept_type)
            case _:
                raise ValueError(f"Unrecognized extractor type: {self.extractor_type}")

    def _build_retriever(self) -> Retriever:
        return Retriever(
            ontology=self.ontology,
            model_name=self.embedding_model,
            top_k=self.top_k,
            min_similarity=self.min_similarity,
        )

    def _build_reranker(self) -> Reranker | None:
        if not self.reranker_enabled:
            return None
        return Reranker(
            llm_client=self.llm_client,
            prompt_config=self._prompt(self.reranker_prompt),
        )

    def update_config(
        self,
        provider: str | None = None,
        model: str | None = None,
        ontology: str | None = None,
        use_reranker: bool | None = None,
        extractor_type: str | None = None,
    ) -> None:
        # Record which components are affected by actual config changes
        rebuild_set = set()
        if provider is not None and provider != self.provider:
            self.provider = provider
            rebuild_set.add("llm")
        if model is not None and model != self.model:
            self.model = model
            rebuild_set.add("llm")
        if ontology is not None and ontology != self.ontology:
            self.ontology = ontology
            rebuild_set.add("retriever")
        if use_reranker is not None and use_reranker != self.reranker_enabled:
            self.reranker_enabled = use_reranker
            rebuild_set.add("reranker")
        if extractor_type is not None and \
                extractor_type.lower().strip() != self.extractor_type.lower().strip():
            self.extractor_type = extractor_type
            rebuild_set.add("extractor")

        # Rebuild affected components
        if "llm" in rebuild_set:
            # llm update means extractor and reranker has to be updated
            rebuild_set.add("extractor")
            rebuild_set.add("reranker")
            # Rebuild llm client
            self.llm_client = create_llm_client(
                provider=self.provider, model=self.model)
        if "extractor" in rebuild_set:
            self.extractor = self._build_extractor()
        if "retriever" in rebuild_set:
            self.retriever = self._build_retriever()
        if "reranker" in rebuild_set:
            self.reranker = self._build_reranker()

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
        """ Performs the bulk of the NER and NEL work, with optional re-ranking """

        if not text or not text.strip():
            return {"text": text, "Concepts": [], "Sentences": []}

        logger.info("Starting RAG grounder pipeline")
        total_start = time.time()
        step_times = {}

        step1_start = time.time()
        extraction_result = self.extractor.extract(text)
        step_times["extraction"] = time.time() - step1_start

        # This will return potentially the same concept multiple times if it appears throughout the document in different ocassions.
        #  Maybe with different supporting evidence, but there could be a the same sentence repeated leading to a duplicate entry
        concepts_raw = extraction_result.get("Concepts", [])
        # Ordered sentences from the extractor (empty for extractors that don't split)
        sentences = extraction_result.get("Sentences", [])
        logger.info(f"Extraction completed in {step_times['extraction']:.2f}s, found {len(concepts_raw)} concept(s)")
        for i, c in enumerate(concepts_raw, 1):
            logger.debug("  Concept %d: %s", i, c.get("Concept", ""))
            for ev in c.get("Supporting_Evidence", []):
                logger.debug("    - %s", ev)

        if not concepts_raw:
            return {"text": text, "Concepts": [], "Sentences": sentences}

        # As a result of above's observation, we could have identical entries here too.
        # The sentence_index/start/end keys are optional metadata (e.g. from the
        # Hunflair extractor); they default to None for extractors that omit them.
        concepts = [
            {
                "Concept": c.get("Concept", ""),
                "supporting_evidence": c.get("Supporting_Evidence", []),
                "matched_terms": [],
                "sentence_index": c.get("sentence_index"),
                "start": c.get("start"),
                "end": c.get("end"),
            }
            for c in concepts_raw
        ]

        step2_start = time.time()
        for concept in tqdm(concepts, desc="Retrieving terms", leave=False, disable=len(concepts) <= 1):
            concept_name = concept["Concept"]
            evidence_text = "\n".join(concept["supporting_evidence"])
            retrieval_text = f"{concept_name}\n\n{evidence_text}" if evidence_text else concept_name
            concept["matched_terms"] = self.retriever.retrieve(retrieval_text)
            logger.debug("  Retrieved %d terms for: %s", len(concept["matched_terms"]), concept_name)
            for j, (term, score) in enumerate(concept["matched_terms"], 1):
                logger.debug("    %d. %s - %s (score: %.3f)", j, term.id, term.name, score)

        step_times["retrieval"] = time.time() - step2_start
        logger.info(f"Retrieval completed in {step_times['retrieval']:.2f}s")

        if self.reranker is not None:
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
                logger.debug("  Reranked %d terms for: %s", len(concept["matched_terms"]), concept_name)
                for j, (term, score) in enumerate(concept["matched_terms"], 1):
                    logger.debug("    %d. %s - %s (score: %.3f)", j, term.id, term.name, score)
            step_times["reranking"] = time.time() - step3_start
            logger.info(f"Re-ranking completed in {step_times['reranking']:.2f}s")
        else:
            step_times["reranking"] = 0.0
            logger.info("Re-ranking disabled, skipping")

        total_time = time.time() - total_start
        logger.info(
            f"Pipeline timing: "
            f"Extraction={step_times['extraction']:.2f}s, "
            f"Retrieval={step_times['retrieval']:.2f}s, "
            f"Re-ranking={step_times['reranking']:.2f}s, "
            f"Total={total_time:.2f}s"
        )

        return {"text": text, "Concepts": concepts, "Sentences": sentences}

    def ground(self, text: str) -> list[ScoredMatch]:
        """ Coerces the result of `process` into GILDA-compatible results """
        
        result = self.process(text)
        matches: list[ScoredMatch] = []
        for concept in result["Concepts"]:
            concept_matches = self._annotation_matches(concept, span_text=concept["Concept"] or text)
            if concept_matches:
                matches.append(concept_matches[0])
        return matches

    def annotate(self, text: str, min_similarity: float = 0.7) -> list[Annotation]:
        """ Runs the NER and NEL pipeline end-to-end and returns a list of Annotation objects which contain
        the extractions, groundings and a map back to the original span within the input text """

        result = self.process(text)
        annotations: list[Annotation] = []
        for concept in result["Concepts"]:
            if not concept["Concept"].strip():
                continue

            matches = self._annotation_matches(concept, span_text=concept["Concept"])
            if not matches:
                continue

            # Depending on the extractor type, we are going to highlight different text spans
            # If the extractor is hunflair, an NER for diseases, we will only highlight the text of the entities
            if isinstance(self.extractor, Hunflair2Extractor):
                spans = find_evidence_spans(text, [concept["Concept"]], min_similarity=min_similarity)
            # If the extractor is an LLM, then, we are going to highlight the supported evidence instead
            else:
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
