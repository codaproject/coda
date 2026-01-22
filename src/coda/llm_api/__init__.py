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

    Auto-detects the provider from the model name if not specified.
    Falls back to OpenAI if detection fails.

    Parameters
    ----------
    provider : str, optional
        Explicit provider name ("openai", "ollama", etc.).
        If not provided, auto-detects from model name.
    model : str, optional
        Model name (e.g., "gpt-4o-mini", "llama3.2").
        Used for auto-detection if provider not specified.
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
    # Auto-detect from model name
    from coda.llm_api import create_llm_client
    client = create_llm_client(model="gpt-4o-mini")
    client = create_llm_client(model="llama3.2")

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
    # Determine provider
    if provider:
        provider = provider.lower()
    else:
        # Auto-detect from model name
        provider = _detect_provider(model)

    # Get adapter class
    adapter_class = _ADAPTERS.get(provider)
    if not adapter_class:
        available = ", ".join(_ADAPTERS.keys())
        raise ValueError(
            f"Unknown provider '{provider}'. Available providers: {available}"
        )

    # Create instance with provided kwargs
    return adapter_class(model=model, **kwargs)


def _detect_provider(model: Optional[str] = None) -> str:
    """
    Auto-detect provider from model name.

    Parameters
    ----------
    model : str, optional
        Model name to analyze.

    Returns
    -------
    str
        Detected provider name ("openai" or "ollama").
        Defaults to "openai" if detection fails.
    """
    if not model:
        # Default to OpenAI if no model specified
        return "openai"

    model_lower = model.lower()

    # OpenAI model patterns
    openai_patterns = ["gpt-", "o1-", "o3-", "dall-e", "whisper", "tts-"]
    if any(model_lower.startswith(pattern) for pattern in openai_patterns):
        return "openai"

    # Ollama model patterns (common local model names)
    ollama_patterns = ["llama", "mistral", "mixtral", "phi", "gemma", "qwen", "codellama"]
    if any(pattern in model_lower for pattern in ollama_patterns):
        return "ollama"

    # Default to OpenAI for unknown models
    return "openai"


__all__ = ["LLMClient", "OpenAIAdapter", "OllamaAdapter", "create_llm_client"]
