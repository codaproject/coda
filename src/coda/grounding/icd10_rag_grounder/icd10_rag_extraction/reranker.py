"""
LLM-based re-ranking of retrieved ICD-10 codes.
"""

import json
from typing import Dict, Any, List, Optional

from coda.llm_api import LLMClient
from .schemas import RERANKING_SCHEMA
from .utils import validate_icd10_code


class CodeReranker:
    """
    Re-rank retrieved ICD-10 codes using LLM reasoning.
    """

    def __init__(
        self,
        llm_client: LLMClient
    ):
        """Initialize code reranker.

        Parameters
        ----------
        llm_client : LLMClient
            LLM client instance for making API calls.
        """
        self.llm_client = llm_client
        self.schema = RERANKING_SCHEMA

    def rerank(
        self,
        disease: str,
        evidence: List[str],
        retrieved_codes: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Re-rank retrieved ICD-10 codes based on disease and evidence.

        Parameters
        ----------
        disease : str
            Disease name.
        evidence : list of str
            List of supporting evidence strings.
        retrieved_codes : list of dict
            List of retrieved codes with similarity scores.
        system_prompt : str, optional
            Optional custom system prompt.

        Returns
        -------
        dict
            Dictionary with 'Reranked ICD-10 Codes' list.
        """
        if not retrieved_codes:
            return {"Reranked ICD-10 Codes": []}

        # Format retrieved codes with similarity scores
        retrieved_codes_formatted = []
        for code_info in retrieved_codes:
            code = code_info.get('code', '')
            name = code_info.get('name', '')
            similarity = code_info.get('similarity', 0.0)
            retrieved_codes_formatted.append(
                f"  - Code: {code}, Name: {name}, Similarity: {similarity:.3f}"
            )

        if system_prompt is None:
            system_prompt = """You are a medical coding expert that re-ranks retrieved ICD-10 codes.

Consider these factors (in order of importance):
1. **Clinical accuracy**: Does the code accurately represent the diagnosed disease?
2. **Evidence alignment**: Does the code match the supporting clinical evidence?
3. **Specificity**: Prefer more specific codes over general ones when appropriate
4. **Retrieval confidence**: Consider the embedding similarity scores (higher = more relevant)

Return ONLY JSON that matches the provided schema, ordered from most to least appropriate."""

        evidence_text = "\n".join(f"  - {e}" for e in evidence) if evidence else "  (No specific evidence provided)"

        user_prompt = f"""Diagnosed disease:
{disease}

Supporting evidence:
{evidence_text}
"""
        user_prompt += f"""
Retrieved ICD-10 candidate codes (from semantic search):
{"\n".join(retrieved_codes_formatted)}

Re-rank these codes based on how well they match the disease and evidence."""

        try:
            # Use LLMClient's call_with_schema method
            response_json = self.llm_client.call_with_schema(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=self.schema,
                schema_name="reranking_icd10",
                max_retries=3,
                retry_delay=1.0
            )
            
            # Check for API failures
            if response_json.get("api_failed", False):
                print("Error: LLM API call failed")
                return {"Reranked ICD-10 Codes": []}

            # Validate structure
            if 'Reranked ICD-10 Codes' not in response_json:
                print("Warning: Invalid reranking response structure")
                return {"Reranked ICD-10 Codes": []}

            # Create mapping from code to similarity score from retrieved_codes
            code_to_similarity = {}
            for retrieved_code in retrieved_codes:
                code = retrieved_code.get('code', '')
                similarity = retrieved_code.get('similarity', 0.0)
                if code:
                    code_to_similarity[code] = similarity

            # Validate codes: must be valid format AND in retrieved set
            validated_codes = []
            for code_info in response_json['Reranked ICD-10 Codes']:
                code = code_info.get('ICD-10 Code', '')
                if not validate_icd10_code(code):
                    print(f"Warning: Invalid ICD-10 code format '{code}' in reranking result")
                    continue
                
                if code not in code_to_similarity:
                    print(f"Warning: Code '{code}' not in retrieved codes set - skipping")
                    continue
                
                # Add similarity score from retrieved_codes
                similarity = code_to_similarity[code]
                code_info['similarity'] = similarity
                validated_codes.append(code_info)

            return {"Reranked ICD-10 Codes": validated_codes}

        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse reranking JSON response: {e}")
            return {"Reranked ICD-10 Codes": []}

        except Exception as e:
            print(f"Error: Failed to rerank codes: {e}")
            return {"Reranked ICD-10 Codes": []}

