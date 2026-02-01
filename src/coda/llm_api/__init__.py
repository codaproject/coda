"""
LLM API adapter module.

Provides a unified interface for LLM API calls with structured JSON schema output,
retry logic, and error handling. Abstracts provider-specific implementations.
"""

from typing import Optional, Dict, Any

from .client import LLMClient
from .openai_adapter import OpenAIAdapter
from .ollama_adapter import OllamaAdapter

# Registry of available adapters
_ADAPTERS: Dict[str, type[LLMClient]] = {
    "openai": OpenAIAdapter,
    "ollama": OllamaAdapter,
}


def create_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """
    Factory function to create an appropriate LLM client adapter.

    Defaults to OpenAI provider if not specified.

    Parameters
    ----------
    provider : str, optional
        Explicit provider name ("openai", "ollama", etc.).
        If not provided, defaults to "openai".
    model : str, optional
        Model name (e.g., "gpt-4o-mini", "llama3.2").
    **kwargs
        Additional arguments passed to the adapter constructor
        (e.g., api_key, base_url, timeout).

    Returns
    -------
    LLMClient
        An instance of the appropriate adapter class.

    Raises
    ------
    ValueError
        If provider is not recognized or required parameters are missing.

    Examples
    --------
    # Default to OpenAI
    from coda.llm_api import create_llm_client
    client = create_llm_client(model="gpt-4o-mini")

    # Explicit provider
    client = create_llm_client(provider="openai", model="gpt-4o-mini", api_key="sk-...")
    client = create_llm_client(provider="ollama", model="llama3.2")

    # With additional parameters
    client = create_llm_client(
        provider="openai",
        model="gpt-4o",
        api_key="sk-...",
        timeout=(60.0, 300.0)
    )
    """
    # Determine provider - default to "openai" if not specified
    if provider:
        provider = provider.lower()
    else:
        provider = "openai"

    # Get adapter class
    adapter_class = _ADAPTERS.get(provider)
    if not adapter_class:
        available = ", ".join(_ADAPTERS.keys())
        raise ValueError(
            f"Unknown provider '{provider}'. Available providers: {available}"
        )

    # Create instance with provided kwargs
    return adapter_class(model=model, **kwargs)


__all__ = ["LLMClient", "OpenAIAdapter", "OllamaAdapter", "create_llm_client"]
