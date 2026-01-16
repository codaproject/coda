"""
LLM-based re-ranking of retrieved ICD-10 codes.

Supports:
- rerank(...): single-mention reranking (legacy)
- rerank_batch(...): batch reranking for ALL mentions in one LLM call (new)
"""

import json
import os
import time
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI

from .schemas import RERANKING_SCHEMA, BATCH_RERANKING_SCHEMA
from .utils import validate_icd10_code

logger = logging.getLogger(__name__)


class CodeReranker:
    """
    Re-rank ICD-10 candidates using LLM reasoning.

    Batch mode is the intended fast path:
      - one LLM call per input text (across all mentions)

    IMPORTANT (closed-set behavior):
      - The LLM is allowed to choose ONLY from:
          (a) retrieved candidate codes
          (b) optional extractor-proposed llm_code (if present)
      - Any other codes returned by the LLM are dropped.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")

        self.client = OpenAI(api_key=api_key, timeout=(60.0, 300.0))
        self.model = model

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

        response_json = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "reranking_icd10_single",
                            "schema": self.schema_single,
                            "strict": True,
                        }
                    },
                )
                response_json = json.loads(response.output_text)
                break
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Reranking attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} reranking attempts failed. Last error: {e}")
                    return {"Reranked ICD-10 Codes": [], "api_failed": True}

        if response_json is None:
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
              - llm_code: str
              - llm_code_name: str
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
                "- For each mention, you MUST ONLY output codes that appear in that mention's candidate list\n"
                "  (and/or its optional extractor ICD-10 hint if provided).\n"
                "- Do NOT introduce any new ICD-10 codes.\n\n"
                "Rules:\n"
                "- Do NOT infer diagnoses beyond what is explicitly stated in the mention/context.\n"
                "- Prefer codes that best match the mention meaning.\n"
                "- Prefer specificity only when supported by text.\n"
                "- Similarity scores are helpful but not authoritative.\n\n"
                "Return ONLY JSON that matches the provided schema."
            )

        # Build prompt blocks
        blocks: List[str] = []
        for it in runnable_items:
            mention_id = str(it.get("mention_id", "")).strip()
            mention = str(it.get("mention", "")).strip()

            context_list = it.get("context", None)
            if isinstance(context_list, list) and context_list:
                context_text = " | ".join(str(x).strip() for x in context_list if str(x).strip())
            else:
                context_text = ""

            llm_code = self._norm_code(it.get("llm_code", ""))
            llm_code_name = str(it.get("llm_code_name", "") or "").strip()

            retrieved_codes = list(it.get("retrieved_codes", []) or [])
            retrieved_codes_sorted = sorted(
                retrieved_codes,
                key=lambda x: float(x.get("similarity", 0.0) or 0.0),
                reverse=True,
            )[: max(1, int(per_mention_top_k))]

            candidates_lines = []
            for c in retrieved_codes_sorted:
                code = self._norm_code(c.get("code", ""))
                name = str(c.get("name", "")).strip()
                sim = c.get("similarity", None)
                try:
                    sim_f = float(sim) if sim is not None else None
                except Exception:
                    sim_f = None

                if sim_f is None:
                    candidates_lines.append(f"- {code} | {name}")
                else:
                    candidates_lines.append(f"- {code} | {name} | sim={sim_f:.3f}")

            llm_hint = ""
            if llm_code and llm_code_name:
                llm_hint = f"LLM_hint: {llm_code} | {llm_code_name}"

            block = (
                f"[MENTION]\n"
                f"id: {mention_id}\n"
                f"text: {mention}\n"
                + (f"context: {context_text}\n" if context_text else "")
                + (f"{llm_hint}\n" if llm_hint else "")
                + "candidates (YOU MUST ONLY CHOOSE FROM THESE):\n"
                + "\n".join(candidates_lines)
            )
            blocks.append(block)

        user_prompt = (
            "Re-rank ICD-10 candidates for each mention below.\n\n"
            "Output requirements:\n"
            "- Provide a result for each mention id.\n"
            "- For each mention id, return an ordered list of codes from best to worst.\n"
            "- DO NOT include codes not present in that mention's candidate list (or its optional LLM_hint).\n\n"
            + "\n\n".join(blocks)
        )

        response_json = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "reranking_icd10_batch",
                            "schema": self.schema_batch,
                            "strict": True,
                        }
                    },
                )
                response_json = json.loads(response.output_text)
                break
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Batch reranking attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} batch reranking attempts failed. Last error: {e}")
                    return {"Mention Rerankings": [], "api_failed": True}

        if response_json is None:
            return {"Mention Rerankings": [], "api_failed": True}

        if "Mention Rerankings" not in response_json or not isinstance(response_json["Mention Rerankings"], list):
            logger.warning("Invalid batch reranking response structure")
            return {"Mention Rerankings": []}

        # Build per-mention similarity lookup from retrieved candidates
        sim_lookup: Dict[str, Dict[str, Any]] = {}
        # Build per-mention allowed set: retrieved + optional llm_code
        allowed_lookup: Dict[str, set] = {}

        for it in runnable_items:
            mention_id = str(it.get("mention_id", "")).strip()
            retrieved_codes = it.get("retrieved_codes", []) or []
            code_to_sim = {
                self._norm_code(c.get("code", "")): c.get("similarity", None)
                for c in retrieved_codes
                if c.get("code")
            }
            sim_lookup[mention_id] = code_to_sim

            allowed = set(code_to_sim.keys())
            llm_code = self._norm_code(it.get("llm_code", ""))
            if llm_code and validate_icd10_code(llm_code, check_existence=True):
                allowed.add(llm_code)

            allowed_lookup[mention_id] = allowed

        # Validate codes & attach similarity/retrieved_from; enforce closed-set
        validated_outputs: List[Dict[str, Any]] = []
        for entry in response_json["Mention Rerankings"]:
            mention_id = str(entry.get("mention_id", "")).strip()
            ranked = entry.get("Reranked ICD-10 Codes", [])

            code_to_sim = sim_lookup.get(mention_id, {})
            allowed_codes = allowed_lookup.get(mention_id, set())

            cleaned_ranked: List[Dict[str, Any]] = []
            if isinstance(ranked, list):
                for code_info in ranked:
                    code = self._norm_code(code_info.get("ICD-10 Code", ""))
                    name = str(code_info.get("ICD-10 Name", "") or "").strip()

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

        # Ensure every input mention_id is present
        present_ids = {x.get("mention_id", "") for x in validated_outputs}
        for it in items:
            mid = str(it.get("mention_id", "")).strip()
            if mid and mid not in present_ids:
                validated_outputs.append({"mention_id": mid, "Reranked ICD-10 Codes": []})

        return {"Mention Rerankings": validated_outputs}
