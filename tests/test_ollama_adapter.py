from types import SimpleNamespace
from coda.llm_api import ollama_adapter


def test_ollama_adapter_uses_configured_base_url(monkeypatch):
    captured = {}

    class DummyClient:
        def __init__(self, *, host, timeout):
            captured["host"] = host
            captured["timeout"] = timeout

        def chat(self, **kwargs):
            captured["chat"] = kwargs
            return SimpleNamespace(
                message=SimpleNamespace(content="response"),
            )

    monkeypatch.setattr(ollama_adapter, "Client", DummyClient)

    adapter = ollama_adapter.OllamaAdapter(
        base_url="http://ollama.internal:11434",
        model="llama3.2",
        timeout=42.0,
    )

    assert adapter.call("test prompt") == "response"
    assert captured["host"] == "http://ollama.internal:11434"
    assert captured["timeout"] == 42.0
    assert captured["chat"]["model"] == "llama3.2"


def test_ollama_adapter_uses_base_url_from_config(monkeypatch):
    from coda.config import reload_settings

    captured = {}

    class DummyClient:
        def __init__(self, *, host, timeout):
            captured["host"] = host

    monkeypatch.setattr(ollama_adapter, "Client", DummyClient)
    monkeypatch.setenv(
        "CODA_LLM__OLLAMA__BASE_URL",
        "http://host.docker.internal:11434",
    )
    reload_settings()
    try:
        ollama_adapter.OllamaAdapter(model="llama3.2")
        assert captured["host"] == "http://host.docker.internal:11434"
    finally:
        monkeypatch.undo()
        reload_settings()


def test_ollama_adapter_passes_num_ctx_and_think(monkeypatch):
    from coda.config import reload_settings

    captured = {}

    class DummyClient:
        def __init__(self, *, host, timeout):
            pass

        def chat(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(message=SimpleNamespace(content='{"a": 1}'))

    monkeypatch.setattr(ollama_adapter, "Client", DummyClient)
    monkeypatch.setenv("CODA_LLM__OLLAMA__NUM_CTX", "8192")
    monkeypatch.setenv("CODA_LLM__OLLAMA__THINK", "false")
    reload_settings()
    try:
        adapter = ollama_adapter.OllamaAdapter(model="qwen3")
        assert adapter.call_with_schema("s", "u", {}, "x") == {"a": 1}
        assert captured["options"]["num_ctx"] == 8192
        assert captured["think"] is False
    finally:
        monkeypatch.undo()
        reload_settings()


def test_ollama_adapter_omits_unset_think(monkeypatch):
    from coda.config import reload_settings

    captured = {}

    class DummyClient:
        def __init__(self, *, host, timeout):
            pass

        def chat(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(message=SimpleNamespace(content="hi"))

    monkeypatch.setattr(ollama_adapter, "Client", DummyClient)
    monkeypatch.setenv("CODA_LLM__OLLAMA__NUM_CTX", "0")
    monkeypatch.setenv("CODA_LLM__OLLAMA__THINK", "")
    reload_settings()
    try:
        ollama_adapter.OllamaAdapter(model="qwen3").call("u")
        assert "think" not in captured
        assert "num_ctx" not in captured["options"]
    finally:
        monkeypatch.undo()
        reload_settings()
