"""
Main pipeline orchestrator for medical coding.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union

from .extractor import BaseExtractor
from .schemas import COD_EVIDENCE_EXTRACTION_SCHEMA
from .retriever import ICD10Retriever
from .reranker import CodeReranker
from .annotator import annotate
from .utils import load_icd10_definitions

from openacme.icd10.generate_embeddings import generate_icd10_embeddings

logger = logging.getLogger(__name__)


class MedCoderPipeline:
    """
    Complete pipeline for extracting mention spans and assigning ICD-10 codes.

    Combines LLM mention extraction, semantic retrieval, and re-ranking.

    Supports extractor outputs in multiple shapes:
      1) Disease-grouped:
         {"Diseases": [{"Disease": str, "Supporting Evidence": [str, ...]}, ...]}
         -> flattened to Mentions where each evidence span becomes a mention.

      2) COD evidence spans (flat):
         {"COD_EVIDENCE_SPANS": [str, ...]}
         -> converted to Mentions format.

      3) Flat mentions (legacy):
         {"Mentions": [{"Mention": str, "ICD10": str}, ...]}
         Note: ICD10 field is optional and typically empty since extractors no longer predict codes.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        retrieval_top_k: int = 10,
        retrieval_min_similarity: float = 0.0,
        extraction_schema: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the medical coding pipeline.

        Parameters
        ----------
        extraction_schema : Dict[str, Any], optional
            JSON schema for extraction. If None, defaults to DISEASE_EXTRACTION_SCHEMA.
            Common options:
            - DISEASE_EXTRACTION_SCHEMA: Hierarchical extraction (diseases with evidence spans)
            - COD_EVIDENCE_EXTRACTION_SCHEMA: Flat extraction (just evidence spans)

        Notes
        -----
        Embeddings are automatically loaded from openacme's default location.
        The pipeline will ensure embeddings exist by calling generate_icd10_embeddings()
        if needed (idempotent operation).
        """
        if extraction_schema is None:
            extraction_schema = COD_EVIDENCE_EXTRACTION_SCHEMA

        # Create extractor with the provided schema
        self.extractor = BaseExtractor(
            schema=extraction_schema,
            api_key=openai_api_key,
            model=openai_model,
        )

        # Ensure embeddings + definitions exist (idempotent; will not regenerate if present)
        generate_icd10_embeddings()
        load_icd10_definitions()

        self.retriever = ICD10Retriever()
        self.reranker = CodeReranker(api_key=openai_api_key, model=openai_model)

        self.retrieval_top_k = retrieval_top_k
        self.retrieval_min_similarity = retrieval_min_similarity

    # ---------------------------------------------------------------------
    # Disease-grouped -> flat mentions
    # ---------------------------------------------------------------------
    def _flatten_diseases_to_mentions(self, extraction_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert disease-grouped extraction output into flat Mentions[].

        Expected disease-grouped shape:
            {"Diseases": [{"Disease": str, "Supporting Evidence": [str, ...]}, ...]}

        Output flat shape:
            [{"Mention": <evidence span>, "ICD10": "", "_disease": <label>}, ...]
        """
        diseases = extraction_result.get("Diseases", [])
        if not isinstance(diseases, list):
            return []

        flat: List[Dict[str, Any]] = []
        for d in diseases:
            if not isinstance(d, dict):
                continue

            disease_label = (d.get("Disease") or "").strip()
            evidence_list = d.get("Supporting Evidence", [])
            if not isinstance(evidence_list, list):
                evidence_list = []

            for ev in evidence_list:
                if not isinstance(ev, str):
                    continue
                ev_clean = ev.strip()
                if not ev_clean:
                    continue

                flat.append(
                    {
                        # retrieval + rerank operate on evidence spans
                        "Mention": ev_clean,

                        # No ICD10 codes from extractors (retrieval + reranking will assign codes)
                        "ICD10": "",

                        # metadata (annotator ignores unknown fields)
                        "_disease": disease_label,
                    }
                )

        return flat

    def _convert_cod_evidence_spans_to_mentions(self, extraction_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert COD_EVIDENCE_SPANS extraction output into flat Mentions[].

        Expected COD evidence spans shape:
            {"COD_EVIDENCE_SPANS": [str, ...]}

        Output flat shape:
            [{"Mention": <evidence span>, "ICD10": ""}, ...]
        """
        evidence_spans = extraction_result.get("COD_EVIDENCE_SPANS", [])
        if not isinstance(evidence_spans, list):
            return []

        flat: List[Dict[str, Any]] = []
        for span in evidence_spans:
            if not isinstance(span, str):
                continue
            span_clean = span.strip()
            if not span_clean:
                continue

            flat.append(
                {
                    "Mention": span_clean,
                    "ICD10": "",  # No ICD10 codes from CodEvidenceExtractor
                }
            )

        return flat

    def process(
        self,
        clinical_descriptions: Union[str, List[str]],
        text_type: str = "clinical_note",
        identifier: Optional[Union[str, List[str]]] = None,
        annotate_evidence: bool = True,
        annotation_min_similarity: float = 0.7,
        top_k: int = 5,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process clinical description(s) through full pipeline.

        Returns
        -------
        dict or list of dict
            If single description: Dictionary with AnnotatedText structure.
            If list of descriptions: List of dictionaries, each with AnnotatedText structure.

        Internal raw_result shape used for annotation:
            {"Mentions": [ { "Mention": str, "ICD10": str (empty, extractors don't predict codes),
                            "retrieved_codes": [...], "reranked_codes": [...], 
                            "reranking_failed": bool }, ... ]}
        """
        is_single = isinstance(clinical_descriptions, str)
        descriptions_list = [clinical_descriptions] if is_single else clinical_descriptions

        # Normalize identifiers
        if identifier is None:
            identifiers_list = [""] * len(descriptions_list) if descriptions_list else []
        elif isinstance(identifier, str):
            identifiers_list = [identifier] * len(descriptions_list) if descriptions_list else []
        else:
            if len(identifier) != len(descriptions_list):
                raise ValueError(
                    f"Number of identifiers ({len(identifier)}) must match "
                    f"number of descriptions ({len(descriptions_list)})"
                )
            identifiers_list = identifier

        # Empty input handling
        if not descriptions_list:
            empty_result = annotate(
                text="",
                text_type=text_type,
                identifier="",
                pipeline_result={"Mentions": []},
                top_k=top_k,
                add_evidence_spans=annotate_evidence,
                min_similarity=annotation_min_similarity,
            )
            empty_dict = empty_result.model_dump()
            empty_dict["_raw_result"] = {"Mentions": []}
            return empty_dict if is_single else []

        logger.info(f"Starting MedCoder pipeline for {len(descriptions_list)} clinical description(s)")

        results: List[Dict[str, Any]] = []

        for idx, (clinical_description, text_id) in enumerate(zip(descriptions_list, identifiers_list), 1):
            step_times: Dict[str, float] = {}
            total_start = time.time()

            if len(descriptions_list) > 1:
                logger.info(f"Processing description {idx}/{len(descriptions_list)}")

            # Step 1: Extract mentions using LLM
            logger.debug("Step 1: Extracting mentions and optional ICD-10 guesses")
            step1_start = time.time()
            extraction_result = self.extractor.extract(clinical_description)
            step1_time = time.time() - step1_start
            step_times["extraction"] = step1_time

            api_failed = extraction_result.get("api_failed", False)

            # Support multiple extractor output shapes:
            # 1) disease-grouped: {"Diseases": [...]}  -> flatten into Mentions
            # 2) COD evidence spans: {"COD_EVIDENCE_SPANS": [...]}  -> convert to Mentions
            # 3) flat mentions (legacy): {"Mentions": [...]}
            if "Diseases" in extraction_result:
                mentions = self._flatten_diseases_to_mentions(extraction_result)
            elif "COD_EVIDENCE_SPANS" in extraction_result:
                mentions = self._convert_cod_evidence_spans_to_mentions(extraction_result)
            elif "Mentions" in extraction_result:
                mentions = extraction_result.get("Mentions", [])
            else:
                mentions = []

            if api_failed:
                logger.error("Extraction API failed after all retries - returning placeholder result for this row")
                failed_result = annotate(
                    text=clinical_description,
                    text_type=text_type,
                    identifier=text_id,
                    pipeline_result={"Mentions": []},
                    top_k=top_k,
                    add_evidence_spans=annotate_evidence,
                    min_similarity=annotation_min_similarity,
                ).model_dump()
                failed_result["_raw_result"] = {"Mentions": [], "api_failed": True}
                results.append(failed_result)
                continue

            logger.info(f"Extraction completed in {step1_time:.2f}s, found {len(mentions)} mention(s)")

            if not mentions:
                logger.warning("No mentions extracted from clinical description")
                empty_result = annotate(
                    text=clinical_description,
                    text_type=text_type,
                    identifier=text_id,
                    pipeline_result={"Mentions": []},
                    top_k=top_k,
                    add_evidence_spans=annotate_evidence,
                    min_similarity=annotation_min_similarity,
                )
                d = empty_result.model_dump()
                d["_raw_result"] = {"Mentions": []}
                results.append(d)
                continue

            # Step 2: Retrieve candidate codes per mention (semantic search)
            logger.debug(f"Step 2: Retrieving top-{self.retrieval_top_k} similar codes for each mention")
            step2_start = time.time()

            for m in mentions:
                mention_text = (m.get("Mention") or "").strip()
                if not mention_text:
                    m["retrieved_codes"] = []
                    continue

                retrieved = self.retriever.retrieve(
                    mention_text,
                    top_k=self.retrieval_top_k,
                    min_similarity=self.retrieval_min_similarity,
                )
                m["retrieved_codes"] = retrieved
                logger.debug(f"Retrieved {len(retrieved)} codes for mention: {mention_text[:80]}")

            step2_time = time.time() - step2_start
            step_times["retrieval"] = step2_time
            logger.info(f"Retrieval completed in {step2_time:.2f}s for {len(mentions)} mention(s)")

            # Step 3: Re-rank codes for ALL mentions using ONE LLM call
            logger.debug("Step 3: Batch re-ranking codes (single call per text)")
            step3_start = time.time()

            # Helper: fallback = similarity-sorted retrieved codes in reranked schema shape
            def _fallback_from_retrieved(retrieved_codes: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
                if not retrieved_codes:
                    return []
                retrieved_sorted = sorted(
                    retrieved_codes,
                    key=lambda x: float(x.get("similarity", 0.0) or 0.0),
                    reverse=True,
                )[: max(1, int(k))]
                out: List[Dict[str, Any]] = []
                for c in retrieved_sorted:
                    out.append(
                        {
                            "ICD-10 Code": (c.get("code") or "").strip().upper(),
                            "ICD-10 Name": c.get("name", "") or "",
                            "similarity": c.get("similarity", None),
                            "retrieved_from": "semantic_embeddings",
                        }
                    )
                return out

            # Assign stable IDs per mention (used to join batch reranking results)
            for i, m in enumerate(mentions):
                m["_mention_id"] = f"m{i}"

            # Build batch payload
            batch_items: List[Dict[str, Any]] = []
            for m in mentions:
                mention_text = (m.get("Mention") or "").strip()

                # Build context (especially for disease-grouped flattening)
                disease_label = (m.get("_disease") or "").strip()
                context: List[str] = []
                if disease_label:
                    context.append(f"Disease/Condition: {disease_label}")
                if mention_text:
                    context.append(f"Evidence: {mention_text}")

                batch_items.append(
                    {
                        "mention_id": m["_mention_id"],
                        "mention": mention_text,
                        "context": context,
                        "retrieved_codes": m.get("retrieved_codes", []),
                    }
                )

            # Single rerank call
            batch_rerank = self.reranker.rerank_batch(items=batch_items)

            # If API failed, fallback to retrieved codes for all mentions
            if batch_rerank.get("api_failed", False):
                logger.warning("Batch reranking API failed - falling back to retrieved codes only for all mentions")
                for m in mentions:
                    m["reranking_failed"] = True
                    m["reranked_codes"] = _fallback_from_retrieved(m.get("retrieved_codes", []) or [], top_k)
            else:
                # Join results back by mention_id
                by_id: Dict[str, List[Dict[str, Any]]] = {}
                for r in batch_rerank.get("Mention Rerankings", []):
                    rid = r.get("mention_id")
                    codes = r.get("Reranked ICD-10 Codes", [])
                    if rid:
                        by_id[str(rid)] = codes if isinstance(codes, list) else []

                # Per-mention: if reranker yields empty (often due to hallucinated codes filtered out),
                # treat as reranking failure and fallback to retrieved codes.
                for m in mentions:
                    rid = str(m.get("_mention_id", ""))
                    reranked_codes = by_id.get(rid, [])

                    if not reranked_codes:
                        m["reranking_failed"] = True
                        m["reranked_codes"] = _fallback_from_retrieved(m.get("retrieved_codes", []) or [], top_k)
                    else:
                        m["reranking_failed"] = False
                        m["reranked_codes"] = reranked_codes

            # Cleanup internal join key (don’t leak into raw_result unless you want it)
            for m in mentions:
                m.pop("_mention_id", None)

            step3_time = time.time() - step3_start
            step_times["reranking"] = step3_time
            logger.info(f"Batch re-ranking completed in {step3_time:.2f}s for {len(mentions)} mention(s)")

            total_time = time.time() - total_start
            step_times["total"] = total_time

            logger.info(
                f"Description {idx} timing breakdown: "
                f"Extraction={step_times['extraction']:.2f}s, "
                f"Retrieval={step_times['retrieval']:.2f}s, "
                f"Re-ranking={step_times['reranking']:.2f}s, "
                f"Total={step_times['total']:.2f}s"
            )

            raw_result = {"Mentions": mentions}

            annotated_text = annotate(
                text=clinical_description,
                text_type=text_type,
                identifier=text_id,
                pipeline_result=raw_result,
                top_k=top_k,
                add_evidence_spans=annotate_evidence,
                min_similarity=annotation_min_similarity,
            )

            result_dict = annotated_text.model_dump()
            result_dict["_raw_result"] = raw_result
            results.append(result_dict)

        logger.info(f"Pipeline completed for {len(descriptions_list)} description(s)")

        if is_single:
            if results:
                return results[0]
            empty_result = annotate(
                text=descriptions_list[0] if descriptions_list else "",
                text_type=text_type,
                identifier=identifiers_list[0] if identifiers_list else "",
                pipeline_result={"Mentions": []},
                top_k=top_k,
                add_evidence_spans=annotate_evidence,
                min_similarity=annotation_min_similarity,
            ).model_dump()
            empty_result["_raw_result"] = {"Mentions": [], "api_failed": True}
            return empty_result

        return results

    def extract_only(self, clinical_description: str) -> Dict[str, Any]:
        """Only perform mention extraction (no retrieval/re-ranking)."""
        return self.extractor.extract(clinical_description)

    def retrieve_only(self, clinical_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Only perform semantic retrieval (no extraction/re-ranking)."""
        return self.retriever.retrieve(
            clinical_text,
            top_k=top_k or self.retrieval_top_k,
            min_similarity=self.retrieval_min_similarity,
        )
