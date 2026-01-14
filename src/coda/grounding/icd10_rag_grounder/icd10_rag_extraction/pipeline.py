"""
Main pipeline orchestrator for medical coding.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from .extractor import DiseaseExtractor
from .retriever import ICD10Retriever
from .reranker import CodeReranker
from .annotator import annotate
from .utils import (
    combine_text_for_retrieval,
    get_icd10_name,
    validate_icd10_code
)

from openacme.icd10.generate_embeddings import generate_icd10_embeddings


# Set up logging
logger = logging.getLogger(__name__)


class MedCoderPipeline:
    """
    Complete pipeline for extracting diseases and assigning ICD-10 codes.

    Combines LLM extraction, semantic retrieval, and re-ranking.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        retrieval_top_k: int = 10,
        retrieval_min_similarity: float = 0.0
    ):
        """Initialize the medical coding pipeline.

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

        Notes
        -----
        Embeddings are automatically loaded from openacme's default location.
        The pipeline will ensure embeddings exist by calling generate_icd10_embeddings()
        if needed (idempotent operation).
        """
        # Initialize components
        self.extractor = DiseaseExtractor(
            api_key=openai_api_key,
            model=openai_model
        )

        # Ensure embeddings are generated (this is idempotent - won't regenerate if they exist)
        generate_icd10_embeddings()

        self.retriever = ICD10Retriever()

        self.reranker = CodeReranker(
            api_key=openai_api_key,
            model=openai_model
        )

        self.retrieval_top_k = retrieval_top_k
        self.retrieval_min_similarity = retrieval_min_similarity

    def process(
        self,
        clinical_descriptions: Union[str, List[str]],
        text_type: str = "clinical_note",
        identifier: Optional[Union[str, List[str]]] = None,
        annotate_evidence: bool = True,
        annotation_min_similarity: float = 0.7,
        top_k: int = 5
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Process clinical description(s) through full pipeline.

        Parameters
        ----------
        clinical_descriptions : str or list of str
            Clinical note(s) or description text(s). Can be a single string
            or a list of strings.
        text_type : str
            Type of text (e.g., 'va_narrative', 'clinical_note'). Defaults to 'clinical_note'.
        identifier : str, optional
            Unique identifier for the text(s). If None and single description,
            will use empty string. If list, should be list of identifiers.
        annotate_evidence : bool
            If True, add character spans for evidence strings. Defaults to True.
        annotation_min_similarity : float
            Minimum similarity threshold for evidence annotation (0.0-1.0).
            Defaults to 0.7.
        top_k : int
            Number of top ranked matches to include per annotation. Defaults to 5.

        Returns
        -------
        dict or list of dict
            If single description: Dictionary with AnnotatedText structure.
            If list of descriptions: List of dictionaries, each with AnnotatedText structure.
        """
        # Normalize input to list
        is_single = isinstance(clinical_descriptions, str)
        descriptions_list = [clinical_descriptions] if is_single else clinical_descriptions
        
        # Normalize identifiers
        if identifier is None:
            identifiers_list = [""] * len(descriptions_list) if descriptions_list else []
        elif isinstance(identifier, str):
            identifiers_list = [identifier] * len(descriptions_list) if descriptions_list else []
        else:  # identifier is List[str]
            if len(identifier) != len(descriptions_list):
                raise ValueError(
                    f"Number of identifiers ({len(identifier)}) must match "
                    f"number of descriptions ({len(descriptions_list)})"
                )
            identifiers_list = identifier

        if not descriptions_list:
            empty_result = annotate(
                text="",
                text_type=text_type,
                identifier="",
                pipeline_result={"Diseases": []},
                top_k=top_k,
                add_evidence_spans=annotate_evidence,
                min_similarity=annotation_min_similarity
            )
            empty_dict = empty_result.model_dump()
            empty_dict['_raw_result'] = {"Diseases": []}
            return empty_dict if is_single else []

        logger.info(f"Starting MedCoder pipeline for {len(descriptions_list)} clinical description(s)")

        results = []

        for idx, (clinical_description, text_id) in enumerate(zip(descriptions_list, identifiers_list), 1):
            step_times = {}
            total_start = time.time()

            if len(descriptions_list) > 1:
                logger.info(f"Processing description {idx}/{len(descriptions_list)}")

            # Step 1: Extract diseases using LLM
            logger.debug("Step 1: Extracting diseases and initial ICD-10 codes")

            step1_start = time.time()
            extraction_result = self.extractor.extract(clinical_description)
            step1_time = time.time() - step1_start
            step_times['extraction'] = step1_time

            diseases = extraction_result.get('Diseases', [])

            logger.info(f"Extraction completed in {step1_time:.2f}s, found {len(diseases)} disease(s)")

            if not diseases:
                logger.warning("No diseases extracted from clinical description")
                # Create empty AnnotatedText
                empty_result = annotate(
                    text=clinical_description,
                    text_type=text_type,
                    identifier=text_id,
                    pipeline_result={"Diseases": []},
                    top_k=top_k,
                    add_evidence_spans=annotate_evidence,
                    min_similarity=annotation_min_similarity
                )
                results.append(empty_result.model_dump())
                continue

            # Step 2: Retrieve additional codes using semantic search
            logger.debug(f"Step 2: Retrieving top-{self.retrieval_top_k} similar codes for each disease")

            step2_start = time.time()

            for disease in diseases:
                disease_name = disease.get('Disease', '')
                evidence = disease.get('Supporting Evidence', [])

                # Combine disease name + evidence for richer retrieval
                retrieval_text = combine_text_for_retrieval(disease_name, evidence)

                # Retrieve codes
                retrieved = self.retriever.retrieve(
                    retrieval_text,
                    top_k=self.retrieval_top_k,
                    min_similarity=self.retrieval_min_similarity
                )

                disease['retrieved_codes'] = retrieved
                logger.debug(f"Retrieved {len(retrieved)} codes for disease: {disease_name}")

            step2_time = time.time() - step2_start
            step_times['retrieval'] = step2_time

            logger.info(f"Retrieval completed in {step2_time:.2f}s for {len(diseases)} disease(s)")

            # Step 3: Re-rank codes using LLM
            logger.debug("Step 3: Re-ranking codes")

            step3_start = time.time()

            for disease in diseases:
                disease_name = disease.get('Disease', '')
                evidence = disease.get('Supporting Evidence', [])
                llm_code_raw = disease.get('ICD10', '')
                
                # Validate LLM code - if invalid, ignore it and use empty string
                # This ensures reranker only uses retrieved codes when LLM code is wrong
                if llm_code_raw and validate_icd10_code(llm_code_raw):
                    llm_code = llm_code_raw
                    llm_code_name = get_icd10_name(llm_code)
                else:
                    # Invalid or empty code - ignore it, reranker will use only retrieved codes
                    if llm_code_raw and not validate_icd10_code(llm_code_raw):
                        logger.debug(f"Ignoring invalid ICD-10 code '{llm_code_raw}' for disease '{disease_name}'")
                    llm_code = ''
                    llm_code_name = ''
                
                retrieved_codes = disease.get('retrieved_codes', [])

                # Re-rank (will ignore llm_code if empty/invalid)
                reranking_result = self.reranker.rerank(
                    disease=disease_name,
                    evidence=evidence,
                    llm_code=llm_code,
                    llm_code_name=llm_code_name,
                    retrieved_codes=retrieved_codes
                )

                disease['reranked_codes'] = reranking_result.get('Reranked ICD-10 Codes', [])
                disease['llm_code_name'] = llm_code_name

                num_reranked = len(disease['reranked_codes'])
                logger.debug(f"Re-ranked {num_reranked} codes for disease: {disease_name}")

            step3_time = time.time() - step3_start
            step_times['reranking'] = step3_time

            logger.info(f"Re-ranking completed in {step3_time:.2f}s for {len(diseases)} disease(s)")

            total_time = time.time() - total_start
            step_times['total'] = total_time

            if len(descriptions_list) > 1:
                logger.info(f"Completed description {idx}/{len(descriptions_list)} in {total_time:.2f}s")

            # Log timing breakdown
            logger.info(
                f"Description {idx} timing breakdown: "
                f"Extraction={step_times['extraction']:.2f}s, "
                f"Retrieval={step_times['retrieval']:.2f}s, "
                f"Re-ranking={step_times['reranking']:.2f}s, "
                f"Total={step_times['total']:.2f}s"
            )

            # Build raw result dict
            raw_result = {"Diseases": diseases}

            # Convert to AnnotatedText format (optionally adds evidence spans)
            annotated_text = annotate(
                text=clinical_description,
                text_type=text_type,
                identifier=text_id,
                pipeline_result=raw_result,
                top_k=top_k,
                add_evidence_spans=annotate_evidence,
                min_similarity=annotation_min_similarity
            )
            # Convert to dict and store raw_result for internal use (e.g., evidence spans)
            result_dict = annotated_text.model_dump()
            result_dict['_raw_result'] = raw_result
            results.append(result_dict)

        logger.info(f"Pipeline completed for {len(descriptions_list)} description(s)")

        # Return single result if single input, list if multiple inputs
        return results[0] if is_single else results

    def extract_only(
        self,
        clinical_description: str
    ) -> Dict[str, Any]:
        """Only perform disease extraction (no retrieval/re-ranking).

        Parameters
        ----------
        clinical_description : str
            Clinical note or description text.

        Returns
        -------
        dict
            Dictionary with extracted diseases.
        """
        return self.extractor.extract(clinical_description)

    def retrieve_only(
        self,
        clinical_text: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Only perform semantic retrieval (no extraction/re-ranking).

        Parameters
        ----------
        clinical_text : str
            Clinical text to search.
        top_k : int, optional
            Number of codes to retrieve. Defaults to pipeline setting.

        Returns
        -------
        list of dict
            List of retrieved codes with similarity scores.
        """
        return self.retriever.retrieve(
            clinical_text,
            top_k=top_k or self.retrieval_top_k,
            min_similarity=self.retrieval_min_similarity
        )

