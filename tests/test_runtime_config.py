import pytest

from coda import runtime_config


RUNTIME_ENV_VARS = (
    "APP_HOST",
    "APP_PORT",
    "INFERENCE_HOST",
    "INFERENCE_PORT",
    "INFERENCE_URL",
    "CODA_INFERENCE_URL",
    "INFERENCE_LLM_PROVIDER",
    "INFERENCE_LLM_MODEL",
    "OLLAMA_BASE_URL",
    "CODA_KG_URL",
    "RAG_LLM_PROVIDER",
    "RAG_LLM_MODEL",
    "RAG_ONTOLOGY",
    "RAG_USE_RERANKER",
)


def test_runtime_config_defaults(monkeypatch):
    for env_var in RUNTIME_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    assert runtime_config.get_app_host() == runtime_config.DEFAULT_APP_HOST
    assert runtime_config.get_app_port() == runtime_config.DEFAULT_APP_PORT
    assert (runtime_config.get_inference_host()
            == runtime_config.DEFAULT_INFERENCE_HOST)
    assert (runtime_config.get_inference_port()
            == runtime_config.DEFAULT_INFERENCE_PORT)
    assert (runtime_config.get_inference_llm_provider()
            == runtime_config.DEFAULT_INFERENCE_LLM_PROVIDER)
    assert (runtime_config.get_inference_llm_model()
            == runtime_config.DEFAULT_INFERENCE_LLM_MODEL)
    assert runtime_config.get_inference_url() == (
        f"http://{runtime_config.DEFAULT_LOCAL_INFERENCE_HOST}:"
        f"{runtime_config.DEFAULT_INFERENCE_PORT}"
    )
    assert (runtime_config.get_ollama_base_url()
            == runtime_config.DEFAULT_OLLAMA_BASE_URL)
    assert runtime_config.get_kg_url() == runtime_config.DEFAULT_KG_URL
    assert (runtime_config.get_rag_llm_provider()
            == runtime_config.DEFAULT_RAG_LLM_PROVIDER)
    assert (runtime_config.get_rag_llm_model()
            == runtime_config.DEFAULT_RAG_LLM_MODEL)
    assert (runtime_config.get_rag_ontology()
            == runtime_config.DEFAULT_RAG_ONTOLOGY)
    assert (runtime_config.get_rag_use_reranker()
            == runtime_config.DEFAULT_RAG_USE_RERANKER)


def test_runtime_config_env_overrides(monkeypatch):
    monkeypatch.setenv("APP_HOST", "127.0.0.1")
    monkeypatch.setenv("APP_PORT", "9000")
    monkeypatch.setenv("INFERENCE_HOST", "127.0.0.1")
    monkeypatch.setenv("INFERENCE_PORT", "6123")
    monkeypatch.setenv("INFERENCE_URL", "http://inference.internal:6123")
    monkeypatch.setenv("INFERENCE_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("INFERENCE_LLM_MODEL", "llama3.2")
    monkeypatch.setenv(
        "OLLAMA_BASE_URL",
        "http://ollama.internal:11434",
    )
    monkeypatch.setenv("CODA_KG_URL", "bolt://kg.internal:7687")
    monkeypatch.setenv("RAG_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("RAG_LLM_MODEL", "llama3.2")
    monkeypatch.setenv("RAG_ONTOLOGY", "icd11")
    monkeypatch.setenv("RAG_USE_RERANKER", "false")

    assert runtime_config.get_app_host() == "127.0.0.1"
    assert runtime_config.get_app_port() == 9000
    assert runtime_config.get_inference_host() == "127.0.0.1"
    assert runtime_config.get_inference_port() == 6123
    assert runtime_config.get_inference_url() == "http://inference.internal:6123"
    assert runtime_config.get_inference_llm_provider() == "ollama"
    assert runtime_config.get_inference_llm_model() == "llama3.2"
    assert (runtime_config.get_ollama_base_url()
            == "http://ollama.internal:11434")
    assert runtime_config.get_kg_url() == "bolt://kg.internal:7687"
    assert runtime_config.get_rag_llm_provider() == "ollama"
    assert runtime_config.get_rag_llm_model() == "llama3.2"
    assert runtime_config.get_rag_ontology() == "icd11"
    assert runtime_config.get_rag_use_reranker() is False


def test_runtime_config_blank_inference_url_falls_back(monkeypatch):
    monkeypatch.setenv("INFERENCE_PORT", "7001")
    monkeypatch.setenv("INFERENCE_URL", "  ")

    assert runtime_config.get_inference_url() == "http://127.0.0.1:7001"


def test_runtime_config_supports_legacy_inference_url(monkeypatch):
    monkeypatch.delenv("INFERENCE_URL", raising=False)
    monkeypatch.setenv(
        "CODA_INFERENCE_URL",
        "http://legacy-inference.internal:5123",
    )

    assert (runtime_config.get_inference_url()
            == "http://legacy-inference.internal:5123")


def test_runtime_config_prefers_canonical_inference_url(monkeypatch):
    monkeypatch.setenv("INFERENCE_URL", "http://inference.internal:5123")
    monkeypatch.setenv(
        "CODA_INFERENCE_URL",
        "http://legacy-inference.internal:5123",
    )

    assert runtime_config.get_inference_url() == "http://inference.internal:5123"


def test_runtime_config_blank_ollama_url_falls_back(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "  ")

    assert (runtime_config.get_ollama_base_url()
            == runtime_config.DEFAULT_OLLAMA_BASE_URL)


def test_runtime_config_rejects_invalid_boolean(monkeypatch):
    monkeypatch.setenv("RAG_USE_RERANKER", "sometimes")

    with pytest.raises(ValueError, match="RAG_USE_RERANKER"):
        runtime_config.get_rag_use_reranker()
