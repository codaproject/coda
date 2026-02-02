"""
Ollama API adapter implementation.

Handles Ollama-specific API calls with structured JSON schema output.
Uses Ollama's native structured output support via the `format` parameter.
"""

import json
import os
import time
import logging
from typing import Dict, Any, Optional

try:
    from ollama import chat
except ImportError:
    chat = None

from .client import LLMClient

logger = logging.getLogger(__name__)


class OllamaAdapter(LLMClient):
    """
    Ollama implementation of LLM client adapter.
    
    Handles Ollama API calls with structured JSON schema output.
    Uses Ollama's native schema-based validation via `format=schema` parameter
    for strict schema enforcement, preventing formatting issues and ensuring
    exact key matching.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "llama3.2",
        timeout: float = 300.0,
    ):
        """
        Initialize Ollama adapter.

        Parameters
        ----------
        base_url : str, optional
            Ollama API base URL. Defaults to http://localhost:11434/api.
        model : str, default="llama3.2"
            Ollama model to use.
        timeout : float, default=300.0
            Request timeout in seconds.
        """
        if chat is None:
            raise ImportError(
                "Ollama package is not installed. "
                "Install it with: pip install 'coda[ollama]' or pip install ollama"
            )
        
        # Note: base_url and timeout are not used with ollama package
        # but kept for API compatibility
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model
        self.timeout = timeout
        self.provider = "ollama"

    def call(self, user_prompt: str, temperature: float = 0.0) -> str:
        """
        Make an Ollama API call without schema constraints.

        Parameters
        ----------
        user_prompt : str
            User prompt for the LLM.
        temperature : float, default=0.0
            Temperature for the LLM. Controls randomness in output.
            Lower values (0.0-0.3) are more deterministic, higher values (0.7-1.0) are more creative.

        Returns
        -------
        str
            Raw text response from the LLM.

        Raises
        ------
        RuntimeError
            If all retry attempts fail or if the response is empty.
        """
        messages = [{"role": "user", "content": user_prompt}]
        last_error = None

        for attempt in range(3):  # Default max_retries
            try:
                response = chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": temperature,
                    },
                )
                
                # Extract message content
                response_text = response.message.content.strip()
                
                if not response_text:
                    raise ValueError("Empty response from Ollama")
                
                return response_text
                
            except Exception as e:
                last_error = e
                if attempt < 2:  # max_retries - 1
                    delay = 1.0 * (2 ** attempt)  # Default retry_delay
                    logger.warning(
                        f"Ollama API call attempt {attempt + 1}/3 failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All 3 Ollama API call attempts failed. Last error: {e}"
                    )

        # If we get here, all retries failed
        raise RuntimeError(
            f"Ollama API call failed after 3 attempts. Last error: {last_error}"
        ) from last_error

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
        Make an Ollama API call with structured JSON schema output.

        Uses Ollama's native schema-based validation via `format=schema` parameter
        for strict schema enforcement. This ensures exact key matching and prevents
        formatting issues like whitespace in keys.

        Parameters
        ----------
        system_prompt : str
            System prompt for the LLM.
        user_prompt : str
            User prompt for the LLM.
        schema : Dict[str, Any]
            JSON schema for structured output.
        schema_name : str
            Name identifier for the schema (used in prompts if needed).
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
        # Build messages for chat endpoint
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        for attempt in range(max_retries):
            try:
                # Use Ollama's chat function with schema-based validation
                # Pass schema directly for strict validation (prevents formatting issues)
                response = chat(
                    model=self.model,
                    messages=messages,
                    format=schema,  # Pass schema directly for strict validation
                    options={
                        "temperature": 0.0,  # Deterministic output
                    },
                )
                
                # Extract message content
                response_text = response.message.content.strip()
                
                if not response_text:
                    raise ValueError("Empty response from Ollama")
                
                # With schema-based validation, response should be clean JSON
                # Parse JSON directly
                response_json = json.loads(response_text)
                
                # Log parsed JSON for debugging
                logger.debug(f"Ollama parsed JSON keys: {list(response_json.keys()) if isinstance(response_json, dict) else 'not a dict'}")
                
                # With proper schema validation, response should match schema exactly
                # But handle edge cases where wrapper might still appear
                if isinstance(response_json, dict):
                    # Check for Ollama's validation wrapper (shouldn't happen with schema validation)
                    if "data" in response_json and any(k in response_json for k in ["membername", "instance", "length"]):
                        logger.debug("Found Ollama wrapper despite schema validation, extracting data")
                        response_json = response_json.get("data", response_json)
                
                return response_json
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    delay = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Ollama API call attempt {attempt + 1}/{max_retries} failed: JSON decode error: {e}. "
                        f"Response: {response_text[:200] if 'response_text' in locals() else 'N/A'}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {max_retries} Ollama API call attempts failed. Last error: JSON decode error: {e}"
                    )
                    return {"api_failed": True}
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Ollama API call attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {max_retries} Ollama API call attempts failed. Last error: {e}"
                    )
                    return {"api_failed": True}

        return {"api_failed": True}

    def get_properties(self) -> Dict[str, Any]:
        """Get metadata properties for Ollama adapter."""
        return {
            "model": self.model,
            "provider": self.provider,
        }
