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
