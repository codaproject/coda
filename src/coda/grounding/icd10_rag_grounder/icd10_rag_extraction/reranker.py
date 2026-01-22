"""
LLM-based re-ranking of retrieved ICD-10 codes.

Supports:
- rerank(...): single-mention reranking (legacy)
- rerank_batch(...): batch reranking for ALL mentions in one LLM call (new)
"""

import logging
from typing import Dict, Any, List, Optional

from coda.llm_api import LLMClient

from .schemas import RERANKING_SCHEMA, BATCH_RERANKING_SCHEMA
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

    # ---------------------------------------------------------------------
    # Legacy: single-mention rerank (kept for compatibility)
    # ---------------------------------------------------------------------
    def rerank(
        self,
        mention: str,
        context: List[str],
        llm_code: str,
        llm_code_name: str,
        retrieved_codes: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Re-rank retrieved ICD-10 codes for a single mention (legacy).
        Prefer rerank_batch for performance.
        """
        if not retrieved_codes:
            return {"Reranked ICD-10 Codes": []}

        # Normalize / build retrieved candidate set
        code_to_similarity: Dict[str, Any] = {}
        retrieved_codes_formatted = []
        for code_info in retrieved_codes:
            code = self._norm_code(code_info.get("code", ""))
            name = str(code_info.get("name", "") or "").strip()
            sim = code_info.get("similarity", None)

            if code:
                code_to_similarity[code] = sim

            retrieved_codes_formatted.append(
                f"  - Code: {code}, Name: {name}, Similarity: {float(sim or 0.0):.3f}"
            )

        # CLOSED-SET: retrieved + optional extractor llm_code
        llm_code_n = self._norm_code(llm_code)
        llm_code_name = str(llm_code_name or "").strip()

        allowed_codes = set(code_to_similarity.keys())
        if llm_code_n and validate_icd10_code(llm_code_n, check_existence=True):
            allowed_codes.add(llm_code_n)

        if system_prompt is None:
            system_prompt = (
                "You are a medical coding expert.\n\n"
                "Your task is to re-rank ICD-10 codes for a single clinical mention.\n\n"
                "CRITICAL CONSTRAINT:\n"
                "- You MUST ONLY output ICD-10 codes that appear in the candidate list below\n"
                "  (and/or the optional initial ICD-10 suggestion if provided).\n"
                "- Do NOT introduce any new ICD-10 codes.\n\n"
                "Consider (in order):\n"
                "1) Mention alignment\n"
                "2) Terminological accuracy (no unstated inference)\n"
                "3) Specificity if justified\n"
                "4) Retrieval similarity\n"
                "5) Initial LLM suggestion only if appropriate\n\n"
                "Return ONLY JSON that matches the provided schema."
            )

        context_text = "\n".join(f"  - {c}" for c in context) if context else "  (No additional context)"

        if llm_code_n and llm_code_name:
            llm_code_section = (
                "\nInitial ICD-10 suggestion:\n"
                f"  Code: {llm_code_n}\n"
                f"  Name: {llm_code_name}\n"
            )
        else:
            llm_code_section = "\n(No initial ICD-10 suggestion provided)\n"

        user_prompt = (
            f"Clinical mention:\n{mention}\n\n"
            f"Context:\n{context_text}\n"
            f"{llm_code_section}\n"
            f"Retrieved ICD-10 candidate codes (YOU MUST ONLY CHOOSE FROM THESE):\n"
            f"{chr(10).join(retrieved_codes_formatted)}\n\n"
            "Re-rank these codes by how well they match the mention."
        )

        # Use LLM adapter for API call
        response_json = self.llm_client.call_with_schema(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=self.schema_single,
            schema_name="reranking_icd10_single",
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )

        # Check for API failure
        if response_json.get("api_failed", False):
            return {"Reranked ICD-10 Codes": [], "api_failed": True}

        if "Reranked ICD-10 Codes" not in response_json:
            logger.warning("Invalid reranking response structure")
            return {"Reranked ICD-10 Codes": []}

        validated_codes: List[Dict[str, Any]] = []
        for code_info in response_json["Reranked ICD-10 Codes"]:
            code = self._norm_code(code_info.get("ICD-10 Code", ""))
            name = str(code_info.get("ICD-10 Name", "") or "").strip()

            if not code:
                continue

            # CLOSED-SET enforcement
            if code not in allowed_codes:
                logger.info(f"Dropping out-of-candidate code '{code}' returned by reranker (closed-set).")
                continue

            # Existence validation (still good hygiene)
            if not validate_icd10_code(code, check_existence=True):
                logger.warning(f"Invalid ICD-10 code '{code}' in reranking output")
                continue

            similarity = code_to_similarity.get(code, None)
            retrieved_from = "semantic_embeddings" if code in code_to_similarity else "llm_generated"

            validated_codes.append(
                {
                    "ICD-10 Code": code,
                    "ICD-10 Name": name,
                    "similarity": similarity,
                    "retrieved_from": retrieved_from,
                }
            )

        return {"Reranked ICD-10 Codes": validated_codes}

    # ---------------------------------------------------------------------
    # New: batch rerank (ONE LLM call for all mentions)
    # ---------------------------------------------------------------------
    def rerank_batch(
        self,
        items: List[Dict[str, Any]],
        *,
        per_mention_top_k: int = 10,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Batch rerank across many mentions in ONE LLM call.

        Parameters
        ----------
        items : list of dict
            Each item MUST include:
              - mention_id: str
              - mention: str
              - retrieved_codes: List[{"code","name","similarity",...}]
            Optional:
              - context: List[str]
        per_mention_top_k : int
            Max number of candidates shown to LLM per mention (controls prompt length).
        system_prompt : str, optional
            Override system prompt.

        Returns
        -------
        dict:
          {
            "Mention Rerankings": [
              {
                "mention_id": str,
                "Reranked ICD-10 Codes": [
                  {"ICD-10 Code": str, "ICD-10 Name": str, "similarity": float|None, "retrieved_from": str},
                  ...
                ]
              },
              ...
            ]
          }
        """
        if not items:
            return {"Mention Rerankings": []}

        # Filter out items with no candidates (LLM has nothing to rerank)
        runnable_items = []
        for it in items:
            retrieved = it.get("retrieved_codes", []) or []
            if retrieved:
                runnable_items.append(it)

        if not runnable_items:
            return {
                "Mention Rerankings": [
                    {"mention_id": str(it.get("mention_id", "")).strip(), "Reranked ICD-10 Codes": []}
                    for it in items
                ]
            }

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

        # Pre-process all items: build sim_lookup, name_lookup, allowed_lookup, and prompt blocks in one pass
        blocks: List[str] = []
        sim_lookup: Dict[str, Dict[str, Any]] = {}
        name_lookup: Dict[str, Dict[str, str]] = {}  # mention_id -> {code: name}
        allowed_lookup: Dict[str, set] = {}
        normalized_mention_ids: Dict[int, str] = {}  # Cache normalized IDs by object id

        for it in runnable_items:
            # Cache normalized mention_id
            mention_id_raw = it.get("mention_id", "")
            mention_id = str(mention_id_raw).strip()
            normalized_mention_ids[id(it)] = mention_id

            mention = str(it.get("mention", "")).strip()

            # Process context
            context_list = it.get("context", None)
            if isinstance(context_list, list) and context_list:
                context_text = " | ".join(str(x).strip() for x in context_list if str(x).strip())
            else:
                context_text = ""

            # Process retrieved_codes once - build prompt and lookup structures
            retrieved_codes = it.get("retrieved_codes", []) or []
            
            # Build code_to_sim, code_to_name, and normalize codes
            code_to_sim: Dict[str, Any] = {}
            code_to_name: Dict[str, str] = {}
            for c in retrieved_codes:
                code_raw = c.get("code", "")
                if code_raw:
                    code = self._norm_code(code_raw)
                    code_to_sim[code] = c.get("similarity", None)
                    code_to_name[code] = str(c.get("name", "") or "").strip()
            
            sim_lookup[mention_id] = code_to_sim
            name_lookup[mention_id] = code_to_name
            allowed_lookup[mention_id] = set(code_to_sim.keys())

            # Sort for prompt (top-k)
            retrieved_codes_sorted = sorted(
                retrieved_codes,
                key=lambda x: self._safe_float(x.get("similarity", 0.0)) or 0.0,
                reverse=True,
            )[: max(1, int(per_mention_top_k))]

            # Build candidate lines for prompt
            candidates_lines = []
            for c in retrieved_codes_sorted:
                code = self._norm_code(c.get("code", ""))
                name = str(c.get("name", "")).strip()
                sim_f = self._safe_float(c.get("similarity"))

                if sim_f is None:
                    candidates_lines.append(f"- {code} | {name}")
                else:
                    candidates_lines.append(f"- {code} | {name} | sim={sim_f:.3f}")

            block = (
                f"[MENTION]\n"
                f"id: {mention_id}\n"
                f"text: {mention}\n"
                + (f"context: {context_text}\n" if context_text else "")
                + "candidates (YOU MUST ONLY CHOOSE FROM THESE):\n"
                + "\n".join(candidates_lines)
            )
            blocks.append(block)

        user_prompt = (
            "Re-rank ICD-10 candidates for each mention below.\n\n"
            "Output requirements:\n"
            "- Provide a result for each mention id.\n"
            "- For each mention id, return an ordered list of ICD-10 CODES ONLY (as strings) from best to worst.\n"
            "- DO NOT include codes not present in that mention's candidate list.\n"
            "- DO NOT include code names - only return the codes themselves.\n\n"
            + "\n\n".join(blocks)
        )

        # Use LLM adapter for API call
        response_json = self.llm_client.call_with_schema(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=self.schema_batch,
            schema_name="reranking_icd10_batch",
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )

        # Check for API failure
        if response_json.get("api_failed", False):
            logger.warning("Batch reranking API call failed")
            return {"Mention Rerankings": [], "api_failed": True}

        # Log the response structure for debugging
        logger.debug(f"Batch reranking response keys: {list(response_json.keys())}")
        logger.debug(f"Response type: {type(response_json)}")
        
        if "Mention Rerankings" not in response_json:
            logger.error(
                f"Invalid batch reranking response structure: missing 'Mention Rerankings' key. "
                f"Got keys: {list(response_json.keys())}"
            )
            logger.error(f"Full response_json (first 1000 chars): {str(response_json)[:1000]}")
            return {"Mention Rerankings": []}
        
        if not isinstance(response_json["Mention Rerankings"], list):
            logger.error(
                f"Invalid batch reranking response structure: 'Mention Rerankings' is not a list. "
                f"Got type: {type(response_json['Mention Rerankings'])}, value: {response_json['Mention Rerankings']}"
            )
            return {"Mention Rerankings": []}

        # Validate codes & attach similarity/retrieved_from; enforce closed-set
        # Note: sim_lookup, name_lookup, and allowed_lookup already built above
        validated_outputs: List[Dict[str, Any]] = []
        present_ids: Dict[str, bool] = {}  # Use dict for O(1) lookup instead of set

        for entry in response_json["Mention Rerankings"]:
            mention_id = str(entry.get("mention_id", "")).strip()
            ranked = entry.get("Reranked ICD-10 Codes", [])

            code_to_sim = sim_lookup.get(mention_id, {})
            code_to_name = name_lookup.get(mention_id, {})
            allowed_codes = allowed_lookup.get(mention_id, set())

            cleaned_ranked: List[Dict[str, Any]] = []
            if isinstance(ranked, list):
                for code_item in ranked:
                    # Handle both old format (dict with "ICD-10 Code") and new format (string)
                    if isinstance(code_item, dict):
                        code = self._norm_code(code_item.get("ICD-10 Code", ""))
                    else:
                        code = self._norm_code(code_item)

                    if not code:
                        continue

                    # CLOSED-SET enforcement
                    if code not in allowed_codes:
                        logger.info(
                            f"Dropping out-of-candidate code '{code}' in batch rerank "
                            f"(mention_id={mention_id}) (closed-set)."
                        )
                        continue

                    if not validate_icd10_code(code, check_existence=True):
                        logger.warning(
                            f"Invalid ICD-10 code '{code}' in batch reranking output (mention_id={mention_id})"
                        )
                        continue

                    # Look up name from retrieved codes
                    name = code_to_name.get(code, "")
                    similarity = code_to_sim.get(code, None)
                    retrieved_from = "semantic_embeddings" if code in code_to_sim else "llm_generated"

                    cleaned_ranked.append(
                        {
                            "ICD-10 Code": code,
                            "ICD-10 Name": name,
                            "similarity": similarity,
                            "retrieved_from": retrieved_from,
                        }
                    )

            validated_outputs.append(
                {
                    "mention_id": mention_id,
                    "Reranked ICD-10 Codes": cleaned_ranked,
                }
            )
            present_ids[mention_id] = True

        # Ensure every input mention_id is present (use cached normalized IDs)
        for it in items:
            mention_id_raw = it.get("mention_id", "")
            mid = normalized_mention_ids.get(id(it), str(mention_id_raw).strip())
            if mid and mid not in present_ids:
                validated_outputs.append({"mention_id": mid, "Reranked ICD-10 Codes": []})

        return {"Mention Rerankings": validated_outputs}
