"""
OpenAI API adapter implementation.

Handles OpenAI-specific API calls with structured JSON schema output.
"""

import json
import os
import time
import logging
from typing import Dict, Any, Optional

from openai import OpenAI

from .client import LLMClient

logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMClient):
    """
    OpenAI implementation of LLM client adapter.
    
    Handles OpenAI API calls with structured JSON schema output,
    retry logic, and error handling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout: tuple = (60.0, 300.0),
    ):
        """
        Initialize OpenAI adapter.

        Parameters
        ----------
        model : str, default="gpt-4o-mini"
            OpenAI model to use.
        api_key : str, optional
            OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
        timeout : tuple, default=(60.0, 300.0)
            Timeout tuple (connect_timeout, read_timeout) in seconds.
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")

        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.provider = "openai"

    def call_with_schema(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
        schema_name: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Make an OpenAI API call with structured JSON schema output.

        Parameters
        ----------
        system_prompt : str
            System prompt for the LLM.
        user_prompt : str
            User prompt for the LLM.
        schema : Dict[str, Any]
            JSON schema for structured output.
        schema_name : str
            Name identifier for the schema (used in API calls).
        max_retries : int, default=3
            Maximum number of retry attempts on failure.
        retry_delay : float, default=1.0
            Base delay in seconds for exponential backoff retries.

        Returns
        -------
        Dict[str, Any]
            Parsed JSON response matching the schema.
            Includes "api_failed": True if all retries failed.
        """
        response_json = None

        for attempt in range(max_retries):
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
                            "name": schema_name,
                            "schema": schema,
                            "strict": True,
                        }
                    },
                )
                response_json = json.loads(response.output_text)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"LLM API call attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {max_retries} LLM API call attempts failed. Last error: {e}"
                    )
                    return {"api_failed": True}

        if response_json is None:
            return {"api_failed": True}

        return response_json

    def get_properties(self) -> Dict[str, Any]:
        """Get metadata properties for OpenAI adapter."""
        return {
            "model": self.model,
            "provider": self.provider,
        }
