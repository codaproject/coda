"""
LLM-based disease extraction from clinical notes.
"""

import json
import os
from typing import Dict, Any, Optional
from openai import OpenAI

from .schemas import DISEASE_EXTRACTION_SCHEMA
from .utils import validate_extraction_result, validate_icd10_code


class DiseaseExtractor:
    """
    Extract diseases, evidence, and initial ICD-10 codes from clinical notes.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini"
    ):
        """Initialize disease extractor.

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
        self.schema = DISEASE_EXTRACTION_SCHEMA
    
    def extract(
        self,
        clinical_description: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract diseases and ICD-10 codes from clinical description.

        Parameters
        ----------
        clinical_description : str
            Clinical note or description text.
        system_prompt : str, optional
            Optional custom system prompt.

        Returns
        -------
        dict
            Dictionary with 'Diseases' list containing disease info.
        """
        if not clinical_description or not clinical_description.strip():
            return {"Diseases": []}
        
        if system_prompt is None:
            system_prompt = (
                "You are a medical coding assistant that extracts diseases and supporting evidence "
                "from clinical descriptions.\n\n"
                "CRITICAL: For 'Supporting Evidence', you MUST extract EXACT verbatim text spans "
                "from the input text. Do NOT paraphrase, reword, or summarize. Copy the text exactly "
                "as it appears in the clinical description.\n\n"
                "Example:\n"
                "Input: 'Patient has chest pain and shortness of breath.'\n"
                "Correct evidence: ['chest pain', 'shortness of breath']\n"
                "WRONG evidence: ['Patient presents with chest discomfort', 'difficulty breathing']\n\n"
                "Provide accurate ICD-10 codes for each identified disease."
            )
        
        try:
            user_prompt = (
                f"Extract diseases and supporting evidence from the following clinical description.\n\n"
                f"IMPORTANT: For 'Supporting Evidence', copy EXACT text spans from the description below. "
                f"Do not paraphrase or reword.\n\n"
                f"Clinical Description:\n{clinical_description}"
            )
            
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "disease_evidence_icd10",
                        "schema": self.schema,
                        "strict": True
                    }
                }
            )
            
            # Parse response
            response_json = json.loads(response.output_text)
            
            # Validate structure
            if not validate_extraction_result(response_json):
                print("Warning: Invalid response structure from LLM")
                return {"Diseases": []}
            
            # Validate ICD-10 codes and evidence
            validated_diseases = []
            clinical_lower = clinical_description.lower()
            
            for disease in response_json.get('Diseases', []):
                code = disease.get('ICD10', '')
                if not validate_icd10_code(code):
                    print(f"Warning: Invalid ICD-10 code '{code}' for disease '{disease.get('Disease', '')}'")
                    continue
                
                # Validate evidence strings are verbatim (case-insensitive check)
                evidence = disease.get('Supporting Evidence', [])
                validated_evidence = []
                
                for ev in evidence:
                    ev_clean = ev.strip()
                    if not ev_clean:
                        continue
                    
                    # Check if evidence is a substring of the input (case-insensitive)
                    ev_lower = ev_clean.lower()
                    if ev_lower in clinical_lower:
                        validated_evidence.append(ev_clean)
                    else:
                        # Try to find fuzzy match - if it's very similar, it might be okay
                        # But log a warning
                        print(f"Warning: Evidence '{ev_clean[:50]}...' may not be verbatim from input text")
                        validated_evidence.append(ev_clean)  # Still include it, but warn
                
                # Update disease with validated evidence
                disease['Supporting Evidence'] = validated_evidence
                validated_diseases.append(disease)
            
            return {"Diseases": validated_diseases}
        
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON response: {e}")
            return {"Diseases": []}
        
        except Exception as e:
            print(f"Error: Failed to extract diseases: {e}")
            return {"Diseases": []}

