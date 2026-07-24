"""Runtime configuration helpers for deployment wiring."""

import os

DEFAULT_APP_HOST = "0.0.0.0"
DEFAULT_APP_PORT = 8000
DEFAULT_INFERENCE_HOST = "0.0.0.0"
DEFAULT_INFERENCE_PORT = 5123
DEFAULT_INFERENCE_LLM_PROVIDER = "openai"
DEFAULT_INFERENCE_LLM_MODEL = "gpt-5.4-mini"
DEFAULT_GROUNDER_TYPE = "gilda"
DEFAULT_LOCAL_INFERENCE_HOST = "127.0.0.1"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_KG_URL = "bolt://localhost:7687"
DEFAULT_RAG_LLM_PROVIDER = "openai"
DEFAULT_RAG_LLM_MODEL = "gpt-4o-mini"
DEFAULT_RAG_ONTOLOGY = "icd10"
DEFAULT_RAG_USE_RERANKER = True
DEFAULT_RAG_EXTRACTOR_TYPE = "hunflair"
DEFAULT_ONBOARDING_NOTICE_ENABLED = False
DEFAULT_ONBOARDING_NOTICE_FILE = ""
DEFAULT_ONBOARDING_NOTICE_VERSION = "default"
DEFAULT_TRANSCRIBER_BACKEND = "whisper-livekit"
DEFAULT_SPEECHMATICS_URL = "wss://us.rt.speechmatics.com/v2/"
DEFAULT_SPEECHMATICS_MODEL = "enhanced"


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    if not value:
        return default
    return int(value)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of: true, false, 1, 0, yes, no, on, off"
    )


def get_app_host() -> str:
    return _get_str("APP_HOST", DEFAULT_APP_HOST)


def get_app_port() -> int:
    return _get_int("APP_PORT", DEFAULT_APP_PORT)


def get_inference_host() -> str:
    return _get_str("INFERENCE_HOST", DEFAULT_INFERENCE_HOST)


def get_inference_port() -> int:
    return _get_int("INFERENCE_PORT", DEFAULT_INFERENCE_PORT)


def get_inference_llm_provider() -> str:
    return _get_str(
        "INFERENCE_LLM_PROVIDER",
        DEFAULT_INFERENCE_LLM_PROVIDER,
    )


def get_inference_llm_model() -> str:
    return _get_str(
        "INFERENCE_LLM_MODEL",
        DEFAULT_INFERENCE_LLM_MODEL,
    )

def get_grounder_type() -> str:
    return _get_str(
        "GROUNDER_TYPE",
        DEFAULT_GROUNDER_TYPE,
    )


def get_inference_url() -> str:
    inference_url = os.getenv("INFERENCE_URL", "").strip()
    if inference_url:
        return inference_url
    return _get_str(
        "CODA_INFERENCE_URL",
        f"http://{DEFAULT_LOCAL_INFERENCE_HOST}:{get_inference_port()}",
    )


def get_ollama_base_url() -> str:
    return _get_str("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)


def get_openai_base_url() -> str | None:
    """OpenAI-compatible endpoint override, or None to use the default."""
    value = (os.getenv("OPENAI_BASE_URL") or "").strip()
    return value or None


def get_openai_api_key() -> str | None:
    """OpenAI API key, or None if unset."""
    value = (os.getenv("OPENAI_API_KEY") or "").strip()
    return value or None


def get_kg_url() -> str:
    return _get_str("CODA_KG_URL", DEFAULT_KG_URL)


def get_rag_llm_provider() -> str:
    return _get_str("RAG_LLM_PROVIDER", DEFAULT_RAG_LLM_PROVIDER)


def get_rag_llm_model() -> str:
    return _get_str("RAG_LLM_MODEL", DEFAULT_RAG_LLM_MODEL)


def get_rag_ontology() -> str:
    return _get_str("RAG_ONTOLOGY", DEFAULT_RAG_ONTOLOGY)


def get_rag_use_reranker() -> bool:
    return _get_bool("RAG_USE_RERANKER", DEFAULT_RAG_USE_RERANKER)


def get_rag_extractor_type() -> str:
    return _get_str("RAG_EXTRACTOR_TYPE", DEFAULT_RAG_EXTRACTOR_TYPE)


def get_onboarding_notice_enabled() -> bool:
    return _get_bool(
        "CODA_ONBOARDING_NOTICE_ENABLED",
        DEFAULT_ONBOARDING_NOTICE_ENABLED,
    )


def get_onboarding_notice_file() -> str:
    return _get_str(
        "CODA_ONBOARDING_NOTICE_FILE",
        DEFAULT_ONBOARDING_NOTICE_FILE,
    )


def get_onboarding_notice_version() -> str:
    return _get_str(
        "CODA_ONBOARDING_NOTICE_VERSION",
        DEFAULT_ONBOARDING_NOTICE_VERSION,
    )


def get_transcriber_backend() -> str:
    return _get_str("TRANSCRIBER_BACKEND", DEFAULT_TRANSCRIBER_BACKEND)


def get_speechmatics_url() -> str:
    return _get_str("SPEECHMATICS_URL", DEFAULT_SPEECHMATICS_URL)


def get_speechmatics_model() -> str:
    return _get_str("SPEECHMATICS_MODEL", DEFAULT_SPEECHMATICS_MODEL)
