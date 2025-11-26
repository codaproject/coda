"""
LLM-based re-ranking of retrieved ICD-10 codes.
"""

import json
import os
from typing import Dict, Any, List, Optional
from openai import OpenAI

from .schemas import RERANKING_SCHEMA
from .utils import validate_icd10_code


class CodeReranker:
    """
    Re-rank retrieved ICD-10 codes using LLM reasoning.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini"
    ):
        """Initialize code reranker.

        Parameters
        ----------
        api_key : str, optional
            OpenAI API key. Defaults to OPENAI_API_KEY environment variable.
        model : str
            OpenAI model name. Defaults to "gpt-4o-mini".
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.schema = RERANKING_SCHEMA
    
    def rerank(
        self,
        disease: str,
        evidence: List[str],
        llm_code: str,
        llm_code_name: str,
        retrieved_codes: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Re-rank retrieved ICD-10 codes based on disease, evidence, and LLM prediction.

        Parameters
        ----------
        disease : str
            Disease name.
        evidence : list of str
            List of supporting evidence strings.
        llm_code : str
            Initial ICD-10 code from LLM.
        llm_code_name : str
            Name corresponding to llm_code.
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
5. **LLM consistency**: How well does the code align with the initial LLM prediction?

Return ONLY JSON that matches the provided schema, ordered from most to least appropriate."""
        
        evidence_text = "\n".join(f"  - {e}" for e in evidence) if evidence else "  (No specific evidence provided)"
        
        user_prompt = f"""Diagnosed disease:
{disease}

Supporting evidence:
{evidence_text}

LLM's initial ICD-10 prediction:
  Code: {llm_code}
  Name: {llm_code_name}

Retrieved ICD-10 candidate codes (from semantic search):
{"\n".join(retrieved_codes_formatted)}

Re-rank these codes based on how well they match the disease and evidence."""
        
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
                        "name": "reranking_icd_10_codes",
                        "schema": self.schema,
                        "strict": True,
                    }
                },
            )
            
            response_json = json.loads(response.output_text)
            
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
            
            # Validate codes and add similarity scores
            validated_codes = []
            for code_info in response_json['Reranked ICD-10 Codes']:
                code = code_info.get('ICD-10 Code', '')
                if validate_icd10_code(code):
                    # Add similarity score from retrieved_codes if available
                    similarity = code_to_similarity.get(code, 0.0)
                    code_info['similarity'] = similarity
                    validated_codes.append(code_info)
                else:
                    print(f"Warning: Invalid ICD-10 code '{code}' in reranking result")
            
            return {"Reranked ICD-10 Codes": validated_codes}
        
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse reranking JSON response: {e}")
            return {"Reranked ICD-10 Codes": []}
        
        except Exception as e:
            print(f"Error: Failed to rerank codes: {e}")
            return {"Reranked ICD-10 Codes": []}

