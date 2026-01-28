"""
Main pipeline orchestrator for medical coding.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union

from coda.llm_api import LLMClient

from .extractor import BaseExtractor
from .schemas import COD_EVIDENCE_EXTRACTION_SCHEMA
from .retriever import ICD10Retriever
from .reranker import CodeReranker
from .annotator import annotate, find_evidence_spans
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

    Examples
    --------
    Using OpenAI:
        from coda.llm_api import create_llm_client
        llm_client = create_llm_client(model="gpt-4o-mini", api_key="sk-...")
        pipeline = MedCoderPipeline(llm_client=llm_client)

    Using Ollama:
        from coda.llm_api import create_llm_client
        llm_client = create_llm_client(model="llama3.2")
        pipeline = MedCoderPipeline(llm_client=llm_client)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        retrieval_top_k: int = 10,
        retrieval_min_similarity: float = 0.0,
        extraction_schema: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the medical coding pipeline.

        Parameters
        ----------
        llm_client : LLMClient
            LLM client adapter to use for extraction and reranking.
        retrieval_top_k : int, default=10
            Number of ICD-10 codes to retrieve per mention.
        retrieval_min_similarity : float, default=0.0
            Minimum similarity threshold for retrieval.
        extraction_schema : Dict[str, Any], optional
            JSON schema for extraction. If None, defaults to COD_EVIDENCE_EXTRACTION_SCHEMA.
            Common options:
            - DISEASE_EXTRACTION_SCHEMA: Hierarchical extraction (diseases with evidence spans)
            - COD_EVIDENCE_EXTRACTION_SCHEMA: Flat extraction (just evidence spans)

        Notes
        -----
        Embeddings are automatically loaded from openacme's default location.
        The pipeline will ensure embeddings exist by calling generate_icd10_embeddings()
        if needed (idempotent operation).

        Examples
        --------
        # Using OpenAI
        from coda.llm_api import create_llm_client
        llm_client = create_llm_client(model="gpt-4o-mini", api_key="sk-...")
        pipeline = MedCoderPipeline(llm_client=llm_client)

        # Using Ollama
        llm_client = create_llm_client(model="llama3.2")
        pipeline = MedCoderPipeline(llm_client=llm_client)
        """
        if extraction_schema is None:
            extraction_schema = COD_EVIDENCE_EXTRACTION_SCHEMA

        # Store LLM client for metadata extraction
        self.llm_client = llm_client
        
        # Get LLM properties from the client itself
        self.llm_properties = llm_client.get_properties()

        # Create extractor with the provided schema and LLM client
        self.extractor = BaseExtractor(
            schema=extraction_schema,
            llm_client=llm_client,
        )

        # Ensure embeddings + definitions exist (idempotent; will not regenerate if present)
        generate_icd10_embeddings()
        load_icd10_definitions()

        self.retriever = ICD10Retriever()
        self.reranker = CodeReranker(llm_client=llm_client)

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
            [{"Mention": <evidence span>, "_disease": <label>}, ...]
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
            [{"Mention": <evidence span>}, ...]
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
                    "Mention": span_clean
                }
            )

        return flat

    def process_raw(
        self,
        clinical_descriptions: Union[str, List[str]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process clinical description(s) through extraction, retrieval, and reranking.
        Returns raw results without annotation.

        Parameters
        ----------
        clinical_descriptions : str or list of str
            Clinical text(s) to process.

        Returns
        -------
        dict or list of dict
            If single description: Dictionary with raw_result structure.
            If list of descriptions: List of dictionaries, each with raw_result structure.
            
            Raw result shape:
            {
                "Mentions": [
                    {
                        "Mention": str,
                        "ICD10": str (empty, extractors don't predict codes),
                        "retrieved_codes": [...],
                        "reranked_codes": [...],
                        "reranking_failed": bool
                    },
                    ...
                ],
                "api_failed": bool (optional)
            }
        
        Note: All retrieved/reranked codes are included in the output (no truncation).
        Use retrieval_top_k parameter in constructor to control how many codes are retrieved.
        """
        is_single = isinstance(clinical_descriptions, str)
        descriptions_list = [clinical_descriptions] if is_single else clinical_descriptions

        # Empty input handling
        if not descriptions_list:
            return {"Mentions": []} if is_single else [{"Mentions": []}]

        logger.info(f"Starting MedCoder pipeline (raw) for {len(descriptions_list)} clinical description(s)")

        results: List[Dict[str, Any]] = []

        for idx, clinical_description in enumerate(descriptions_list, 1):
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
            logger.info(f"Extraction result keys: {list(extraction_result.keys())}")
            logger.info(f"Extraction result sample (first 500 chars): {str(extraction_result)[:500]}")
            
            if "Diseases" in extraction_result:
                logger.info("Using disease-grouped extraction format")
                mentions = self._flatten_diseases_to_mentions(extraction_result)
            elif "COD_EVIDENCE_SPANS" in extraction_result:
                logger.info(f"Using COD_EVIDENCE_SPANS extraction format. Found {len(extraction_result.get('COD_EVIDENCE_SPANS', []))} spans")
                mentions = self._convert_cod_evidence_spans_to_mentions(extraction_result)
                logger.info(f"Converted to {len(mentions)} mentions")
            elif "Mentions" in extraction_result:
                logger.info(f"Using flat Mentions extraction format. Found {len(extraction_result.get('Mentions', []))} mentions")
                mentions = extraction_result.get("Mentions", [])
            else:
                logger.warning(f"Unknown extraction result format. Keys: {list(extraction_result.keys())}")
                logger.warning(f"Full extraction_result: {extraction_result}")
                mentions = []

            if api_failed:
                logger.error("Extraction API failed after all retries - returning placeholder result for this row")
                results.append({"Mentions": [], "api_failed": True})
                continue

            logger.info(f"Extraction completed in {step1_time:.2f}s, found {len(mentions)} mention(s)")

            if not mentions:
                logger.warning("No mentions extracted from clinical description")
                results.append({"Mentions": []})
                continue

            # Validate mentions using find_evidence_spans - filter out unmatched spans
            validated_mentions = []
            for m in mentions:
                mention_text = (m.get("Mention") or "").strip()
                if not mention_text:
                    continue
                
                # Use find_evidence_spans to check if mention can be matched
                spans = find_evidence_spans(
                    clinical_description,
                    [mention_text],
                    min_similarity=0.7,  # Use default similarity threshold
                    case_sensitive=False,
                )
                
                if spans and spans[0].get("match_type") in ("exact", "fuzzy"):
                    # Mention can be matched, keep it
                    validated_mentions.append(m)
                else:
                    # Mention cannot be matched, filter it out
                    logger.warning(
                        f"Filtering out mention '{mention_text[:60]}...' "
                        f"(match_type: {spans[0].get('match_type') if spans else 'not_found'})"
                    )
            
            mentions = validated_mentions
            logger.info(f"After validation: {len(mentions)} mention(s) remain")

            if not mentions:
                logger.warning("No valid mentions after validation - all were filtered out")
                results.append({"Mentions": []})
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
            def _fallback_from_retrieved(retrieved_codes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """Fallback: return all retrieved codes sorted by similarity (best to worst)."""
                if not retrieved_codes:
                    return []
                retrieved_sorted = sorted(
                    retrieved_codes,
                    key=lambda x: float(x.get("similarity", 0.0) or 0.0),
                    reverse=True,
                )
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
            # Initialize reranking flags for all mentions
            for i, m in enumerate(mentions):
                m["_mention_id"] = str(i)
                m["reranking_failed"] = False
                m["reranked_codes"] = []

            # Build batch payload
            batch_items: List[Dict[str, Any]] = []
            for m in mentions:
                mention_text = (m.get("Mention") or "").strip()
                batch_items.append(
                    {
                        "mention_id": m["_mention_id"],
                        "mention": mention_text,
                        "retrieved_codes": m.get("retrieved_codes", []),
                    }
                )

            # Single rerank call
            logger.debug(f"Batch items (input) for reranking step: {batch_items}")
            batch_rerank = self.reranker.rerank_batch(items=batch_items)

            # If API failed, fallback to retrieved codes for all mentions
            if batch_rerank.get("api_failed", False):
                logger.warning("Batch reranking API failed - falling back to similarity-based ranking for codes for all mentions")
                for m in mentions:
                    m["reranking_failed"] = True
                    m["reranked_codes"] = _fallback_from_retrieved(m.get("retrieved_codes", []) or [])
            else:
                # Join results back by mention_id
                by_id: Dict[str, List[Dict[str, Any]]] = {}
                for r in batch_rerank.get("Mention Rerankings", []):
                    rid = r.get("mention_id")
                    codes = r.get("Reranked ICD-10 Codes", [])
                    if rid:
                        by_id[str(rid)] = codes if isinstance(codes, list) else []

                # For each mention, if reranker yields empty (often due to hallucinated codes filtered out),
                # treat as reranking failure and fallback to retrieved codes.
                for m in mentions:
                    rid = str(m.get("_mention_id", ""))
                    reranked_codes = by_id.get(rid, [])

                    if not reranked_codes:
                        mention_text = (m.get("Mention") or "").strip()[:80]
                        logger.debug(
                            f"Reranker returned empty codes for mention '{mention_text}' "
                            f"(mention_id={rid}) - falling back to similarity-based ranking"
                        )
                        m["reranking_failed"] = True
                        m["reranked_codes"] = _fallback_from_retrieved(m.get("retrieved_codes", []) or [])
                    else:
                        m["reranking_failed"] = False
                        m["reranked_codes"] = reranked_codes
                        logger.debug(
                            f"Reranker returned {len(reranked_codes)} codes for mention_id={rid}"
                        )

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
            results.append(raw_result)

        logger.info(f"Pipeline (raw) completed for {len(descriptions_list)} description(s)")

        return results[0] if is_single else results

    def process(
        self,
        clinical_descriptions: Union[str, List[str]],
        text_type: str = "clinical_note",
        identifier: Optional[str] = None,
        annotate_evidence: bool = True,
        annotation_min_similarity: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Process clinical description(s) through full pipeline with annotation.

        Parameters
        ----------
        clinical_descriptions : str or list of str
            Clinical text(s) to process. If a list is provided, each string is processed
            separately, then results are merged. The final text field will be the
            concatenated original texts.
        text_type : str, default="clinical_note"
            Type of text being processed.
        identifier : str, optional
            Identifier for the text. If None, uses empty string.
        annotate_evidence : bool, default=True
            Whether to add evidence spans to annotations.
        annotation_min_similarity : float, default=0.7
            Minimum similarity threshold for annotation.
        
        Returns
        -------
        dict
            Dictionary with AnnotatedText structure containing all annotations from
            all processed descriptions, with text field being the concatenated original texts.
        
        Note: All retrieved/reranked codes are included in the output (no truncation).
        Use retrieval_top_k parameter in constructor to control how many codes are retrieved.
        """
        # Normalize to list for iteration
        if isinstance(clinical_descriptions, str):
            descriptions_list = [clinical_descriptions]
        else:
            descriptions_list = [d for d in clinical_descriptions if d and str(d).strip()]

        if not descriptions_list:
            empty_result = annotate(
                text="",
                text_type=text_type,
                identifier=identifier or "",
                pipeline_result={"Mentions": []},
                add_evidence_spans=annotate_evidence,
                min_similarity=annotation_min_similarity,
                properties=self.llm_properties,
            ).model_dump()
            return empty_result

        # Process each description separately
        all_mentions: List[Dict[str, Any]] = []
        for description in descriptions_list:
            if not description or not str(description).strip():
                continue
            
            raw_result = self.process_raw(description)
            if isinstance(raw_result, dict) and "Mentions" in raw_result:
                mentions = raw_result["Mentions"]
                all_mentions.extend(mentions)

        # Concatenate original texts for final output
        combined_text = " ".join(str(d).strip() for d in descriptions_list if d and str(d).strip())
        
        # Combine all mentions into a single result
        combined_raw_result = {"Mentions": all_mentions}
        
        # Annotate the combined result using the concatenated text
        annotated_text = annotate(
            text=combined_text,
            text_type=text_type,
            identifier=identifier or "",
            pipeline_result=combined_raw_result,
            add_evidence_spans=annotate_evidence,
            min_similarity=annotation_min_similarity,
            properties=self.llm_properties,
        )
        
        return annotated_text.model_dump()

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
