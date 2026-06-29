"""OpenAI API adapter implementations.

Two adapters share one interface and differ only in which API they use for
structured output: OpenAIChatAdapter (Chat Completions, the default and the
portable choice for OpenAI-compatible servers such as mlx_lm.server, vLLM, or
Ollama's compat layer) and OpenAIResponsesAdapter (the OpenAI Responses API,
opt-in for its newer-API features).
"""

import json
import time
import logging
from typing import Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from coda.runtime_config import get_openai_api_key, get_openai_base_url
from .client import LLMClient

logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMClient):
    """Shared base for OpenAI-compatible adapters.

    Holds client construction, the plain `call`, retry/backoff, and metadata.
    Subclasses implement `_structured_request`, the only place the Responses and
    Chat Completions APIs differ.
    """

    provider = "openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5.4-mini",
        timeout: tuple = (60.0, 300.0),
        base_url: Optional[str] = None,
    ):
        """Initialize OpenAI adapter.

        Parameters
        ----------
        model : str, default="gpt-5.4-mini"
            OpenAI model to use.
        api_key : str, optional
            OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
        timeout : tuple, default=(60.0, 300.0)
            Timeout tuple (connect_timeout, read_timeout) in seconds.
        base_url : str, optional
            OpenAI-compatible endpoint. Defaults to OPENAI_BASE_URL env var.
            Point this at a local server (e.g. mlx_lm.server or LM Studio) to
            run an Apple Silicon MLX model through the OpenAI interface.
        """
        if OpenAI is None:
            raise ImportError(
                "OpenAI package is not installed. "
                "Install it with: pip install 'coda[openai]' or pip install openai"
            )

        base_url = base_url or get_openai_base_url()
        # Local OpenAI-compatible servers ignore the key but the client still
        # requires a non-empty one.
        api_key = api_key or get_openai_api_key() or \
                  ("local" if base_url else None)
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env "
                             "var or pass api_key.")

        self.client = OpenAI(api_key=api_key, timeout=timeout,
                             base_url=base_url)
        self.model = model
        self.base_url = base_url

    def call(self, user_prompt: str, temperature: float = 0.0) -> str:
        """Make an OpenAI API call without schema constraints.

        Parameters
        ----------
        user_prompt : str
            User prompt for the LLM.
        temperature : float, default=0.0
            Temperature for the LLM.

        Returns
        -------
        str
            Raw text response from the LLM.

        Raises
        ------
        RuntimeError
            If all retry attempts fail or if the response is empty.
        """
        last_error = None

        for attempt in range(3):  # Default max_retries
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                )

                # Extract message content
                if not response.choices:
                    raise ValueError("Empty choices in OpenAI response")

                response_text = response.choices[0].message.content
                if response_text is None:
                    raise ValueError("Empty content in OpenAI response")

                response_text = response_text.strip()
                if not response_text:
                    raise ValueError("Empty response from OpenAI")

                return response_text

            except Exception as e:
                last_error = e
                if attempt < 2:  # max_retries - 1
                    delay = 1.0 * (2 ** attempt)  # Default retry_delay
                    logger.warning(
                        f"OpenAI API call attempt {attempt + 1}/3 failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All 3 OpenAI API call attempts failed. Last error: {e}"
                    )

        # If we get here, all retries failed
        raise RuntimeError(
            f"OpenAI API call failed after 3 attempts. Last error: {last_error}"
        ) from last_error

    def _structured_request(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
        schema_name: str,
        temperature: float,
    ) -> Dict[str, Any]:
        """Issue one structured-output request and return the parsed JSON."""
        raise NotImplementedError

    def call_with_schema(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
        schema_name: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Make an OpenAI API call with structured JSON schema output.

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
        temperature: float, default=0.0
            Temperature for the LLM.

        Returns
        -------
        Dict[str, Any]
            Parsed JSON response matching the schema.
            Includes "api_failed": True if all retries failed.
        """
        response_json = None

        for attempt in range(max_retries):
            try:
                response_json = self._structured_request(
                    system_prompt, user_prompt, schema, schema_name, temperature
                )
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


class OpenAIResponsesAdapter(OpenAIAdapter):
    """Structured output via the OpenAI Responses API (`/v1/responses`)."""

    provider = "openai-responses"

    def _structured_request(self, system_prompt, user_prompt, schema,
                            schema_name, temperature):
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
            temperature=temperature,
        )
        return json.loads(response.output_text)


class OpenAIChatAdapter(OpenAIAdapter):
    """Structured output via Chat Completions (`/v1/chat/completions`).

    The default OpenAI adapter and the portable choice for OpenAI-compatible
    servers that don't implement the Responses API.
    """

    provider = "openai"

    def _structured_request(self, system_prompt, user_prompt, schema,
                            schema_name, temperature):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            },
            temperature=temperature,
        )
        return json.loads(response.choices[0].message.content)
