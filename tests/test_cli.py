"""Lightweight CLI tests.

These exercise the CLI's text pipeline and output plumbing using the toy
inference agent and a stub grounder, so they need no Whisper model, LLM,
Gilda terms, knowledge graph, or network - safe to run on CI.
"""
import asyncio
import json
import sys

import pytest

from coda import cli
from coda.inference.agent import CodaToyInferenceAgent


class _StubGrounder:
    """Grounder that produces no annotations (avoids loading Gilda/KG)."""
    def annotate(self, text):
        return []


class _StubAnnotation:
    """Minimal gilda-compatible annotation for write_outputs."""
    def to_json(self):
        return {"text": "fever"}


def test_run_text_with_toy_agent():
    grounder = _StubGrounder()
    agent = CodaToyInferenceAgent()
    full_text, per_chunk = asyncio.run(
        cli.run_text(grounder, agent, "The child had a high fever and chills.")
    )
    assert full_text.startswith("The child")
    assert len(per_chunk) == 1

    chunk_id, text, annotations, inference, timings = per_chunk[0]
    assert chunk_id == "chunk-0"
    assert inference["causes"]  # toy agent always returns causes
    assert "grounding_s" in timings and "inference_s" in timings


def test_write_outputs(tmp_path):
    per_chunk = [(
        "chunk-0",
        "some narrative text",
        [_StubAnnotation()],
        {
            "causes": {"icd10:R99": {"name": "Other", "identifiers": {"icd10": "R99"},
                                     "score": 1.0}},
            "reasoning": "because",
            "timestamp": 123.0,
            "chunks_processed": 1,
        },
        {"grounding_s": 0.1, "inference_s": 0.2},
    )]
    meta = {"input": "x.txt", "input_type": "text", "mode": "text",
            "agent": "toy", "grounder": "stub"}

    cli.write_outputs(tmp_path, "some narrative text", per_chunk, meta)

    assert (tmp_path / "transcript.txt").read_text().strip() == "some narrative text"
    assert json.loads((tmp_path / "annotations.json").read_text()) == [{"text": "fever"}]
    assert sum(1 for _ in (tmp_path / "chunks.jsonl").open()) == 1

    inference = json.loads((tmp_path / "inference.json").read_text())
    assert "icd10:R99" in inference["causes"]
    assert inference["input_type"] == "text"
    assert inference["timing"]["grounding_s"] == 0.1
    assert inference["timing"]["inference_s"] == 0.2
    assert inference["timing"]["total_s"] == pytest.approx(0.3)


def test_requires_an_input_source(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["coda.cli", "--output", "out"])
    with pytest.raises(SystemExit):
        cli.main()


def test_input_and_text_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["coda.cli", "--input", "a.wav", "--text", "b.txt", "--output", "out"])
    with pytest.raises(SystemExit):
        cli.main()
