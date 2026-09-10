"""Tests for dialogue embedding and the embedding inference plumbing."""
import importlib.util

import numpy as np
import pytest

from coda.embedding import BaseEmbedder
from coda.inference.embedding_agent import EmbeddingCODAgent

HAS_ST = importlib.util.find_spec("sentence_transformers") is not None


class DummyEmbedder(BaseEmbedder):
    """Deterministic embedder for testing plumbing without heavy deps."""
    dim = 2

    def embed(self, text):
        return self.embed_batch([text])[0]

    def embed_batch(self, texts):
        # Encode by length so the vector changes as the transcript grows.
        return np.array([[float(len(t or "")), 1.0] for t in texts],
                        dtype=np.float32)


class DummyModel:
    classes_ = np.array(["A", "B"])

    def predict_proba(self, X):
        return np.tile([0.7, 0.3], (len(X), 1))


class RecordingModel(DummyModel):
    """Records the feature vector it was scored on, for shape/value assertions."""

    def predict_proba(self, X):
        self.last_X = np.asarray(X)
        return super().predict_proba(X)


def _agent():
    return EmbeddingCODAgent(embedder=DummyEmbedder(),
                             models={"generic": DummyModel()})


class TestEmbeddingPlumbing:
    @pytest.mark.asyncio
    async def test_running_embedding_updates(self):
        agent = _agent()
        assert agent.dialogue_embedding is None
        await agent.process_chunk("c1", "fever", [])
        first = agent.dialogue_embedding.copy()
        assert first is not None and first.shape == (2,)
        await agent.process_chunk("c2", "and chest pain", [])
        # Accumulated transcript grew, so the length-based embedding grew.
        assert agent.dialogue_embedding[0] > first[0]

    @pytest.mark.asyncio
    async def test_reset_clears_embedding(self):
        agent = _agent()
        await agent.process_chunk("c1", "fever", [])
        assert agent.dialogue_embedding is not None
        agent.reset()
        assert agent.dialogue_embedding is None

    @pytest.mark.asyncio
    async def test_scaffold_no_models_no_embedding(self):
        # No models -> placeholder, and no embedding computed.
        agent = EmbeddingCODAgent(embedder=DummyEmbedder(), models={})
        r = await agent.process_chunk("c1", "fever", [])
        assert agent.dialogue_embedding is None
        assert "icd10:R99" in r["causes"]


class TestAgeFeature:
    @pytest.mark.asyncio
    async def test_age_appended_when_configured(self):
        from coda.metadata import Metadata

        model = RecordingModel()
        agent = EmbeddingCODAgent(
            embedder=DummyEmbedder(), models={"generic": model},
            feature_config={"use_age": True, "max_age_days": 1826.25})
        agent.metadata = Metadata.from_dict(
            {"profile": {"age": {"value": 2, "unit": "years"}}})
        await agent.process_chunk("c1", "fever", [])

        assert model.last_X.shape == (1, 2 + 3)  # DummyEmbedder.dim + age feature
        assert model.last_X[0, -2:].tolist() == [1.0, 0.0]  # age known, not stillbirth

    @pytest.mark.asyncio
    async def test_age_unknown_when_not_provided(self):
        model = RecordingModel()
        agent = EmbeddingCODAgent(
            embedder=DummyEmbedder(), models={"generic": model},
            feature_config={"use_age": True, "max_age_days": 1826.25})
        await agent.process_chunk("c1", "fever", [])

        assert model.last_X[0, -3:].tolist() == [0.0, 0.0, 0.0]

    @pytest.mark.asyncio
    async def test_stillbirth_flag(self):
        from coda.metadata import Metadata

        model = RecordingModel()
        agent = EmbeddingCODAgent(
            embedder=DummyEmbedder(), models={"generic": model},
            feature_config={"use_age": True, "max_age_days": 1826.25})
        agent.metadata = Metadata.from_dict({"profile": {"stillbirth": True}})
        await agent.process_chunk("c1", "fever", [])

        assert model.last_X[0, -3:].tolist() == [0.0, 0.0, 1.0]

    @pytest.mark.asyncio
    async def test_no_age_feature_when_not_configured(self):
        model = RecordingModel()
        agent = EmbeddingCODAgent(embedder=DummyEmbedder(), models={"generic": model})
        await agent.process_chunk("c1", "fever", [])

        assert model.last_X.shape == (1, 2)


@pytest.mark.skipif(not HAS_ST, reason="sentence-transformers not installed")
class TestSentenceTransformerEmbedder:
    def test_embed_shape_and_norm(self):
        from coda.embedding import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder()
        v = emb.embed("child with fever and cough")
        assert v.shape == (emb.dim,)
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-3)

    def test_empty_text_is_zero(self):
        from coda.embedding import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder()
        v = emb.embed("")
        assert v.shape == (emb.dim,)
        assert np.allclose(v, 0.0)
