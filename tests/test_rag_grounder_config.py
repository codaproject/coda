import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace

neo4j = ModuleType("neo4j")
neo4j.GraphDatabase = SimpleNamespace(driver=lambda *args, **kwargs: object())
sys.modules.setdefault("neo4j", neo4j)

sentence_transformers = ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", sentence_transformers)

from coda.grounding.rag_grounder import grounder as grounder_module

_ORIGINAL_DEFAULT_CONFIG = grounder_module.RAGGrounderConfig.default()


def _config_with_overrides(**kwargs):
    config = deepcopy(_ORIGINAL_DEFAULT_CONFIG)
    config.llm.provider = kwargs.pop("provider", config.llm.provider)
    config.llm.model = kwargs.pop("model", config.llm.model)
    config.retriever.ontology = kwargs.pop(
        "ontology",
        config.retriever.ontology,
    )
    config.reranker.enabled = kwargs.pop(
        "use_reranker",
        config.reranker.enabled,
    )
    config.extractor.type = kwargs.pop(
        "extractor_type",
        config.extractor.type,
    )
    if kwargs:
        raise TypeError(f"Unexpected config override(s): {sorted(kwargs)}")
    return config


def _make_grounder(monkeypatch, **kwargs):
    calls = {
        "llm": [],
        "extractor": 0,
        "retriever": 0,
        "reranker": 0,
    }

    def fake_create_llm_client(*, provider, model):
        calls["llm"].append((provider, model))
        return SimpleNamespace(provider=provider, model=model)

    def fake_build_extractor(self):
        calls["extractor"] += 1
        return SimpleNamespace(kind="extractor", n=calls["extractor"])

    def fake_build_retriever(self):
        calls["retriever"] += 1
        return SimpleNamespace(kind="retriever", n=calls["retriever"])

    def fake_build_reranker(self):
        calls["reranker"] += 1
        if not self.config.reranker.enabled:
            return None
        return SimpleNamespace(kind="reranker", n=calls["reranker"])

    monkeypatch.setattr(
        grounder_module,
        "create_llm_client",
        fake_create_llm_client,
    )
    monkeypatch.setattr(
        grounder_module.RagGrounder,
        "_build_extractor",
        fake_build_extractor,
    )
    monkeypatch.setattr(
        grounder_module.RagGrounder,
        "_build_retriever",
        fake_build_retriever,
    )
    monkeypatch.setattr(
        grounder_module.RagGrounder,
        "_build_reranker",
        fake_build_reranker,
    )
    monkeypatch.setattr(
        grounder_module.RAGGrounderConfig,
        "default",
        classmethod(lambda cls: _config_with_overrides(**kwargs)),
    )

    grounder = grounder_module.RagGrounder()
    return grounder, calls


def test_rag_grounder_creates_llm_client_from_config(monkeypatch):
    captured = {}

    def fake_create_llm_client(*, provider, model):
        captured["provider"] = provider
        captured["model"] = model
        return object()

    monkeypatch.setattr(
        grounder_module,
        "create_llm_client",
        fake_create_llm_client,
    )
    monkeypatch.setattr(
        grounder_module.RagGrounder,
        "_build_extractor",
        lambda self: object(),
    )
    monkeypatch.setattr(
        grounder_module.RagGrounder,
        "_build_retriever",
        lambda self: object(),
    )
    monkeypatch.setattr(
        grounder_module.RagGrounder,
        "_build_reranker",
        lambda self: None,  # Return None because use_reranker=False
    )

    new_config = {
        "provider": "ollama",
        "model": "new-test-model",
        "ontology": "icd11",
        "use_reranker": False,
        "extractor_type": "llm",
    }
    monkeypatch.setattr(
        grounder_module.RAGGrounderConfig,
        "default",
        classmethod(
            lambda cls: _config_with_overrides(**new_config)
        ),
    )

    grounder = grounder_module.RagGrounder()

    assert captured == {
        "provider": new_config["provider"],
        "model": new_config["model"],
    }
    assert grounder.config.llm.provider == new_config["provider"]
    assert grounder.config.llm.model == new_config["model"]
    assert grounder.config.retriever.ontology == new_config["ontology"]
    assert grounder.config.reranker.enabled is new_config["use_reranker"]
    assert grounder.config.extractor.type == new_config["extractor_type"]


def test_update_config_noop_does_not_rebuild_components(monkeypatch):
    grounder, calls = _make_grounder(monkeypatch)
    initial_llm = grounder.llm_client
    initial_provider = initial_llm.provider
    initial_model = initial_llm.model
    initial_extractor = grounder.extractor
    initial_retriever = grounder.retriever
    initial_reranker = grounder.reranker

    grounder.update_config(
        provider=initial_provider,
        model=initial_model,
        ontology=grounder.config.retriever.ontology,
        use_reranker=grounder.config.reranker.enabled,
        extractor_type=" {} ".format(grounder.config.extractor.type.upper()),
    )

    assert calls == {
        "llm": [(initial_provider, initial_model)],
        "extractor": 1,
        "retriever": 1,
        "reranker": 1,
    }
    assert grounder.llm_client is initial_llm
    assert grounder.extractor is initial_extractor
    assert grounder.retriever is initial_retriever
    assert grounder.reranker is initial_reranker


def test_update_config_provider_rebuilds_llm_dependents_only(monkeypatch):
    grounder, calls = _make_grounder(monkeypatch)
    initial_llm = grounder.llm_client
    initial_provider = initial_llm.provider
    initial_model = initial_llm.model
    initial_retriever = grounder.retriever

    new_provider = "test-provider"
    grounder.update_config(provider=new_provider)

    assert calls == {
        "llm": [
            (initial_provider, initial_model),
            (new_provider, initial_model),
        ],
        "extractor": 2,
        "retriever": 1,
        "reranker": 2,
    }
    assert grounder.config.llm.provider != initial_llm.provider
    assert grounder.config.llm.model == initial_llm.model
    assert grounder.llm_client is not initial_llm
    assert grounder.llm_client.provider == new_provider
    assert grounder.llm_client.model == initial_model
    assert grounder.retriever is initial_retriever


def test_update_config_model_rebuilds_llm_dependents_only(monkeypatch):
    grounder, calls = _make_grounder(monkeypatch)
    initial_llm = grounder.llm_client
    initial_provider = initial_llm.provider
    initial_model = initial_llm.model
    initial_retriever = grounder.retriever

    new_model = "new-test-model"
    grounder.update_config(model=new_model)

    assert calls == {
        "llm": [
            (initial_provider, initial_model),
            (initial_provider, new_model),
        ],
        "extractor": 2,
        "retriever": 1,
        "reranker": 2,
    }
    assert grounder.llm_client is not initial_llm
    assert grounder.config.llm.provider == initial_llm.provider
    assert grounder.config.llm.model != initial_llm.model
    assert grounder.llm_client.model == new_model
    assert grounder.retriever is initial_retriever


def test_update_config_ontology_rebuilds_retriever_only(monkeypatch):
    grounder, calls = _make_grounder(monkeypatch)
    initial_llm = grounder.llm_client
    initial_provider = initial_llm.provider
    initial_model = initial_llm.model
    initial_extractor = grounder.extractor
    initial_reranker = grounder.reranker
    initial_retriever = grounder.retriever

    new_ontology = "icd11"
    grounder.update_config(ontology=new_ontology)

    assert calls == {
        "llm": [(initial_provider, initial_model)],
        "extractor": 1,
        "retriever": 2,
        "reranker": 1,
    }
    assert grounder.retriever is not initial_retriever
    assert grounder.config.retriever.ontology == new_ontology
    assert grounder.llm_client is initial_llm
    assert grounder.extractor is initial_extractor
    assert grounder.reranker is initial_reranker


def test_update_config_reranker_toggle_rebuilds_reranker_only(monkeypatch):
    grounder, calls = _make_grounder(monkeypatch, use_reranker=True)
    initial_llm = grounder.llm_client
    initial_provider = initial_llm.provider
    initial_model = initial_llm.model
    initial_extractor = grounder.extractor
    initial_retriever = grounder.retriever

    grounder.update_config(use_reranker=False)

    assert calls == {
        "llm": [(initial_provider, initial_model)],
        "extractor": 1,
        "retriever": 1,
        "reranker": 2,
    }
    assert grounder.config.reranker.enabled is False
    assert grounder.reranker is None
    assert grounder.llm_client is initial_llm
    assert grounder.extractor is initial_extractor
    assert grounder.retriever is initial_retriever


def test_update_config_extractor_type_rebuilds_only_on_normalized_change(monkeypatch):
    grounder, calls = _make_grounder(monkeypatch, extractor_type="llm")
    initial_llm = grounder.llm_client
    initial_provider = initial_llm.provider
    initial_model = initial_llm.model
    initial_retriever = grounder.retriever
    initial_reranker = grounder.reranker
    initial_extractor = grounder.extractor

    grounder.update_config(extractor_type=" LLM ")

    assert grounder.config.extractor.type == "llm"
    assert calls["extractor"] == 1
    assert grounder.extractor.n == 1

    new_extractor_type = "hunflair"
    grounder.update_config(extractor_type=new_extractor_type)

    assert calls == {
        "llm": [(initial_provider, initial_model)],
        "extractor": 2,
        "retriever": 1,
        "reranker": 1,
    }
    assert grounder.extractor is not initial_extractor
    assert grounder.config.extractor.type == new_extractor_type
    assert grounder.llm_client is initial_llm
    assert grounder.retriever is initial_retriever
    assert grounder.reranker is initial_reranker
