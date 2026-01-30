"""
LLM-based re-ranking of retrieved ICD-10 codes.

Supports:
- rerank(...): single-mention reranking (legacy)
- rerank_batch(...): batch reranking for ALL mentions in one LLM call (new)
"""

import logging
from typing import Dict, Any, List, Optional

from coda.llm_api import LLMClient

from .schemas import RERANKING_SCHEMA, BATCH_RERANKING_SCHEMA, get_schema_name
from .utils import validate_icd10_code

logger = logging.getLogger(__name__)


class CodeReranker:
    """
    Re-rank ICD-10 candidates using LLM reasoning.

    Batch mode is the intended fast path:
      - one LLM call per input text (across all mentions)

    IMPORTANT (closed-set behavior):
      - The LLM is allowed to choose ONLY from retrieved candidate codes
      - Any other codes returned by the LLM are dropped.
    """

    def __init__(
        self,
        llm_client: LLMClient,
    ):
        """
        Initialize code reranker.

        Parameters
        ----------
        llm_client : LLMClient
            LLM client adapter to use. Must be provided.
        """
        self.llm_client = llm_client

        self.schema_single = RERANKING_SCHEMA
        self.schema_batch = BATCH_RERANKING_SCHEMA

        self.max_retries = 3
        self.retry_delay = 1.0

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _norm_code(code: Any) -> str:
        return str(code or "").strip().upper()

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Safely convert value to float, returning None if conversion fails."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def rerank_batch(
        self,
        items: List[Dict[str, Any]],
        *,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Batch rerank across many mentions in ONE LLM call.

        Requirements for each item:
        - mention_id: str
        - mention: str
        - retrieved_codes: List[{"code","name","similarity",...}]
        Optional:
        - context: List[str]
        """
        if not items:
            return {"Mention Rerankings": []}

        # Only run the LLM on items that actually have candidates
        runnable_items = [it for it in items if (it.get("retrieved_codes") or [])]
        if not runnable_items:
            raise ValueError(f"No runnable items provided: {items}")

        if system_prompt is None:
            system_prompt = (
                "You are a medical coding expert.\n\n"
                "Task: for EACH clinical mention, re-rank the provided ICD-10 candidate codes.\n\n"
                "CRITICAL CONSTRAINT:\n"
                "- For each mention, you MUST ONLY output codes that appear in that mention's candidate list.\n"
                "- Do NOT introduce any new ICD-10 codes.\n\n"
                "Rules:\n"
                "- Do NOT infer diagnoses beyond what is explicitly stated in the mention/context.\n"
                "- Prefer codes that best match the mention meaning.\n"
                "- Prefer specificity only when supported by text.\n"
                "- Similarity scores are helpful but not authoritative.\n\n"
                "Return ONLY JSON that matches the provided schema."
            )

        # Build prompt blocks + lookups
        blocks: List[str] = []

        # mention_id -> {CODE: similarity}
        sim_lookup: Dict[str, Dict[str, Any]] = {}
        # mention_id -> {CODE: name}
        name_lookup: Dict[str, Dict[str, str]] = {}
        # mention_id -> set(CODE)
        allowed_lookup: Dict[str, set] = {}

        # Keep stable set of expected IDs
        expected_ids = [str(it.get("mention_id", "")).strip() for it in runnable_items]
        expected_id_set = set(expected_ids)

        for it in runnable_items:
            mention_id = str(it.get("mention_id", "")).strip()
            mention = str(it.get("mention", "")).strip()
            retrieved_codes = it.get("retrieved_codes", []) or []
            
            # Populate lookups for validation and output enrichment
            code_to_sim: Dict[str, Any] = {}
            code_to_name: Dict[str, str] = {}
            for c in retrieved_codes:
                code_raw = c.get("code", "")
                if not code_raw:
                    continue
                code = self._norm_code(code_raw)
                code_to_sim[code] = c.get("similarity", None)
                code_to_name[code] = str(c.get("name", "") or "").strip()
            
            sim_lookup[mention_id] = code_to_sim
            name_lookup[mention_id] = code_to_name
            allowed_lookup[mention_id] = set(code_to_sim.keys())
            
            # Build prompt block
            candidates_lines: List[str] = []
            for c in retrieved_codes:
                code = self._norm_code(c.get("code", ""))
                name = str(c.get("name", "") or "").strip()
                if code:
                    candidates_lines.append(f"[CODE]: {code}, [NAME]: {name}")
            
            blocks.append(
                f"[MENTION]\n"
                f"id: {mention_id}\n"
                f"text: {mention}\n"
                + "candidates (YOU MUST ONLY CHOOSE FROM THESE):\n"
                + "\n".join(candidates_lines)
            )

        user_prompt = (
            "Re-rank ICD-10 candidates for each mention below.\n\n"
            "Output requirements:\n"
            "- Provide a result for each mention id.\n"
            "- CRITICAL: Use the EXACT 'id' value from each [MENTION] block as the 'mention_id' in your response.\n"
            "- For each mention id, return an ordered list of ICD-10 CODES ONLY (as strings) from best to worst.\n"
            "- DO NOT include codes not present in that mention's candidate list.\n\n"
            + "\n\n".join(blocks)
        )

        # Get schema name from registry
        schema_name = get_schema_name(self.schema_batch)
        
        response_json = self.llm_client.call_with_schema(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=self.schema_batch,
            schema_name=schema_name,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )
        logger.debug(f"Batch reranking JSON response from LLM: {response_json}")

        if response_json.get("api_failed", False):
            logger.warning("Batch reranking API call failed")
            return {"Mention Rerankings": [], "api_failed": True}

        entries = response_json.get("Mention Rerankings", None)
        if not isinstance(entries, list):
            logger.error(
                "Invalid batch reranking response structure: 'Mention Rerankings' missing or not a list. "
                f"Got type={type(entries)} keys={list(response_json.keys())}"
            )
            return {"Mention Rerankings": []}

        validated_outputs: List[Dict[str, Any]] = []
        present_ids: set = set()

        for entry in entries:
            mention_id = str(entry.get("mention_id", "")).strip()
            ranked = entry.get("Reranked ICD-10 Codes", [])

            # Hard fail / warn if LLM returned an unexpected id
            if mention_id not in expected_id_set:
                logger.warning(
                    f"LLM returned unexpected mention_id '{mention_id}'. "
                    f"Expected one of: {sorted(expected_id_set)[:10]}{'...' if len(expected_id_set) > 10 else ''}. "
                    "Dropping this entry."
                )
                continue

            code_to_sim = sim_lookup.get(mention_id, {})
            code_to_name = name_lookup.get(mention_id, {})
            allowed_codes = allowed_lookup.get(mention_id, set())

            cleaned_ranked: List[Dict[str, Any]] = []
            if isinstance(ranked, list):
                for code_item in ranked:
                    code = self._norm_code(code_item.get("ICD-10 Code", "")) if isinstance(code_item, dict) else self._norm_code(code_item)
                    if not code:
                        continue

                    if code not in allowed_codes:
                        logger.info(
                            f"Dropping out-of-candidate code '{code}' in batch rerank "
                            f"(mention_id={mention_id}) (closed-set)."
                        )
                        continue

                    if not validate_icd10_code(code, check_existence=True):
                        logger.warning(f"Invalid ICD-10 code '{code}' (mention_id={mention_id})")
                        continue

                    cleaned_ranked.append(
                        {
                            "ICD-10 Code": code,
                            "ICD-10 Name": code_to_name.get(code, ""),
                            "similarity": code_to_sim.get(code, None),
                            "retrieved_from": "semantic_embeddings",
                        }
                    )

            validated_outputs.append({"mention_id": mention_id, "Reranked ICD-10 Codes": cleaned_ranked})
            present_ids.add(mention_id)

        # Ensure every runnable mention_id has an output entry
        for mid in expected_ids:
            if mid and mid not in present_ids:
                validated_outputs.append({"mention_id": mid, "Reranked ICD-10 Codes": []})

        return {"Mention Rerankings": validated_outputs}
