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


def test_ollama_adapter_uses_base_url_from_env(monkeypatch):
    captured = {}

    class DummyClient:
        def __init__(self, *, host, timeout):
            captured["host"] = host

    monkeypatch.setattr(ollama_adapter, "Client", DummyClient)
    monkeypatch.setenv(
        "OLLAMA_BASE_URL",
        "http://host.docker.internal:11434",
    )

    ollama_adapter.OllamaAdapter(model="llama3.2")

    assert captured["host"] == "http://host.docker.internal:11434"
