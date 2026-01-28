"""
Abstract LLM client interface.

Defines the protocol for LLM API clients.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class LLMClient(ABC):
    """
    Abstract base class for LLM API clients.
    
    Provides a unified interface for structured JSON schema calls with retry logic.
    """

    @abstractmethod
    def call(self, user_prompt: str) -> str:
        """
        Make an LLM API call with no schema.

        Parameters
        ----------
        user_prompt : str
            User prompt for the LLM.

        Returns
        -------
        str
            Raw text response from the LLM.
            Returns empty string if all retries failed.
        """
        pass

    @abstractmethod
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
        Make an LLM API call with structured JSON schema output.

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

        Raises
        ------
        ValueError
            If API key is not configured.
        """
        pass

    def get_properties(self) -> Dict[str, Any]:
        """
        Get metadata properties for this LLM client.

        Returns a dictionary with information about the LLM configuration,
        such as model name, provider, etc. Used for annotating outputs
        with LLM metadata.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing LLM metadata properties.
            Common keys: 'model', 'provider'
        """
        return {}
