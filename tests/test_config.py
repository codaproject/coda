"""Tests for the unified Dynaconf-backed configuration (coda.config)."""

import os

import pytest

from coda.config import PROMPTS, inference_url, settings

# CODA_-prefixed env vars that override config values.
CODA_ENV_VARS = (
    "CODA_APP__HOST",
    "CODA_APP__PORT",
    "CODA_INFERENCE__HOST",
    "CODA_INFERENCE__PORT",
    "CODA_INFERENCE__URL",
    "CODA_INFERENCE__LLM__PROVIDER",
    "CODA_INFERENCE__LLM__MODEL",
    "CODA_LLM__OLLAMA__BASE_URL",
    "CODA_KG__URL",
    "CODA_GROUNDER__TYPE",
    "CODA_GROUNDER__RAG__LLM__PROVIDER",
    "CODA_GROUNDER__RAG__LLM__MODEL",
    "CODA_GROUNDER__RAG__RETRIEVER__ONTOLOGY",
    "CODA_GROUNDER__RAG__RERANKER__ENABLED",
    "CODA_GROUNDER__RAG__EXTRACTOR__TYPE",
    "CODA_DIALOGUE__TRANSCRIBER_BACKEND",
    "CODA_DIALOGUE__SPEECHMATICS__URL",
    "CODA_DIALOGUE__SPEECHMATICS__MODEL",
)


@pytest.fixture(autouse=True)
def isolate_settings():
    """Reload settings from a clean environment before and after each test.

    ``settings`` is a process-wide singleton, so tests that inject env-var
    overrides must not leak into one another.
    """
    def clean_reload():
        for var in CODA_ENV_VARS:
            os.environ.pop(var, None)
        settings.reload()

    clean_reload()
    yield
    clean_reload()


def test_defaults():
    assert settings.app.host == "0.0.0.0"
    assert settings.app.port == 8000
    assert settings.inference.host == "0.0.0.0"
    assert settings.inference.port == 5123
    assert settings.inference.llm.provider == "openai"
    assert settings.inference.llm.model == "gpt-5.4-mini"
    assert settings.llm.ollama.base_url == "http://localhost:11434"
    assert settings.kg.url == "bolt://localhost:7687"
    assert settings.grounder.type == "gilda"
    assert settings.grounder.rag.llm.provider == "openai"
    assert settings.grounder.rag.llm.model == "gpt-4o-mini"
    assert settings.grounder.rag.retriever.ontology == "icd10"
    assert settings.grounder.rag.retriever.embedding_model == "all-MiniLM-L6-v2"
    assert settings.grounder.rag.retriever.top_k == 10
    assert settings.grounder.rag.retriever.min_similarity == 0.0
    assert settings.grounder.rag.reranker.enabled is True
    assert settings.grounder.rag.extractor.type == "hunflair"
    assert settings.dialogue.transcriber_backend == "whisper-livekit"
    assert settings.dialogue.speechmatics.url == "wss://us.rt.speechmatics.com/v2/"
    assert settings.dialogue.speechmatics.model == "enhanced"


def test_types_are_coerced():
    assert isinstance(settings.app.port, int)
    assert isinstance(settings.inference.port, int)
    assert isinstance(settings.grounder.rag.retriever.top_k, int)
    assert isinstance(settings.grounder.rag.retriever.min_similarity, float)
    assert isinstance(settings.grounder.rag.reranker.enabled, bool)


def test_inference_url_derived_from_host_port():
    assert inference_url() == "http://127.0.0.1:5123"


def test_inference_url_explicit(monkeypatch):
    monkeypatch.setenv("CODA_INFERENCE__URL", "http://inference:5123")
    settings.reload()
    assert inference_url() == "http://inference:5123"


def test_env_var_overrides(monkeypatch):
    monkeypatch.setenv("CODA_APP__PORT", "9000")
    monkeypatch.setenv("CODA_KG__URL", "bolt://kg.internal:7687")
    monkeypatch.setenv("CODA_INFERENCE__LLM__MODEL", "gpt-5.4")
    monkeypatch.setenv("CODA_GROUNDER__RAG__RETRIEVER__ONTOLOGY", "icd11")
    monkeypatch.setenv("CODA_GROUNDER__RAG__RERANKER__ENABLED", "false")
    settings.reload()

    assert settings.app.port == 9000
    assert isinstance(settings.app.port, int)
    assert settings.kg.url == "bolt://kg.internal:7687"
    assert settings.inference.llm.model == "gpt-5.4"
    assert settings.grounder.rag.retriever.ontology == "icd11"
    assert settings.grounder.rag.reranker.enabled is False


def test_secrets_merge_without_clobbering_settings():
    # .secrets.yaml contributes llm.openai.api_key; it must not replace the
    # whole `llm` block (base_url / ollama) defined in settings.yaml.
    assert settings.llm.openai.base_url == ""
    assert settings.llm.ollama.base_url == "http://localhost:11434"


def test_prompts_loaded():
    assert set(PROMPTS) == {
        "extractor_default",
        "extractor_medcoder",
        "reranker_default",
    }
    # The prompt referenced by the default extractor config exists and is usable.
    key = settings.grounder.rag.extractor.prompt
    assert key in PROMPTS
    assert "system_prompt" in PROMPTS[key]
    assert "schema" in PROMPTS[key]
    # Regression: the medcoder prompt must use `supporting_evidence_key`.
    assert "supporting_evidence_key" in PROMPTS["extractor_medcoder"]
    assert "supporting_evidence_field" not in PROMPTS["extractor_medcoder"]
