import asyncio
from types import SimpleNamespace

import httpx
import pytest

from coda.app import server


class DummyWebSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, data):
        self.messages.append(data)


@pytest.mark.asyncio
async def test_first_transcript_starts_inference_immediately(monkeypatch):
    websocket = DummyWebSocket()
    coordinator = server.InferenceSessionCoordinator(websocket, session_id="session-a")
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def fake_process_inference(session, request_generation, batch, buffer_wait_s):
        calls.append((session.session_id, request_generation, batch, buffer_wait_s))
        started.set()
        await release.wait()

    monkeypatch.setattr(server, "process_inference", fake_process_inference)

    await coordinator.enqueue("chunk-1", 1.0, "first text", ["ann-1"])
    await asyncio.wait_for(started.wait(), timeout=1.0)

    assert len(calls) == 1
    assert calls[0][0] == "session-a"
    assert calls[0][2].text == "first text"
    release.set()
    await coordinator.wait_for_idle()


@pytest.mark.asyncio
async def test_coalesces_chunks_arriving_during_inference(monkeypatch):
    websocket = DummyWebSocket()
    coordinator = server.InferenceSessionCoordinator(websocket, session_id="session-a")
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    first_release = asyncio.Event()
    second_release = asyncio.Event()
    calls = []
    active = 0
    max_active = 0

    async def fake_process_inference(session, request_generation, batch, buffer_wait_s):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        calls.append((session.session_id, request_generation, batch, buffer_wait_s))
        if len(calls) == 1:
            first_started.set()
            await first_release.wait()
        elif len(calls) == 2:
            second_started.set()
            await second_release.wait()
        active -= 1

    monkeypatch.setattr(server, "process_inference", fake_process_inference)

    await coordinator.enqueue("chunk-1", 1.0, "alpha", ["a1"])
    await asyncio.wait_for(first_started.wait(), timeout=1.0)
    await coordinator.enqueue("chunk-2", 2.0, "bravo", ["b1"])
    await coordinator.enqueue("chunk-3", 3.0, "charlie", ["c1", "c2"])
    await asyncio.sleep(0)

    assert len(calls) == 1

    first_release.set()
    await asyncio.wait_for(second_started.wait(), timeout=1.0)

    second_batch = calls[1][2]
    assert second_batch.chunk_count == 2
    assert second_batch.text == "bravo charlie"
    assert second_batch.annotations == ["b1", "c1", "c2"]
    assert second_batch.chunk_id == "chunk-3"
    assert second_batch.timestamp == 3.0
    assert calls[1][3] >= 0
    assert max_active == 1

    second_release.set()
    await coordinator.wait_for_idle()


@pytest.mark.asyncio
async def test_new_chunks_during_merged_request_are_drained_afterward(monkeypatch):
    websocket = DummyWebSocket()
    coordinator = server.InferenceSessionCoordinator(websocket, session_id="session-a")
    started = [asyncio.Event(), asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event(), asyncio.Event()]
    calls = []

    async def fake_process_inference(session, request_generation, batch, buffer_wait_s):
        idx = len(calls)
        calls.append((session.session_id, request_generation, batch, buffer_wait_s))
        started[idx].set()
        await release[idx].wait()

    monkeypatch.setattr(server, "process_inference", fake_process_inference)

    await coordinator.enqueue("chunk-1", 1.0, "one", ["a1"])
    await asyncio.wait_for(started[0].wait(), timeout=1.0)

    await coordinator.enqueue("chunk-2", 2.0, "two", ["b1"])
    await coordinator.enqueue("chunk-3", 3.0, "three", ["c1"])
    release[0].set()
    await asyncio.wait_for(started[1].wait(), timeout=1.0)

    await coordinator.enqueue("chunk-4", 4.0, "four", ["d1"])
    release[1].set()
    await asyncio.wait_for(started[2].wait(), timeout=1.0)

    assert [call[2].text for call in calls] == ["one", "two three", "four"]
    assert calls[1][2].annotations == ["b1", "c1"]
    assert calls[2][2].annotations == ["d1"]

    release[2].set()
    await coordinator.wait_for_idle()


@pytest.mark.asyncio
async def test_reset_clears_buffered_work_and_suppresses_stale_results(monkeypatch):
    websocket = DummyWebSocket()
    coordinator = server.InferenceSessionCoordinator(websocket, session_id="session-a")
    infer_started = asyncio.Event()
    infer_release = asyncio.Event()
    infer_calls = []
    reset_calls = []

    async def fake_process_inference(session, request_generation, batch, buffer_wait_s):
        infer_calls.append((request_generation, batch.text))
        infer_started.set()
        await infer_release.wait()
        if await session.is_current_generation(request_generation):
            await session.websocket.send_json(
                {"type": "inference", "chunk_id": batch.chunk_id}
            )

    async def fake_reset(session_id, generation):
        reset_calls.append({"session_id": session_id, "session_generation": generation})

    monkeypatch.setattr(server, "process_inference", fake_process_inference)
    monkeypatch.setattr(server, "_reset_inference_session", fake_reset)

    await coordinator.enqueue("chunk-1", 1.0, "alpha", [])
    await asyncio.wait_for(infer_started.wait(), timeout=1.0)
    await coordinator.enqueue("chunk-2", 2.0, "bravo", [])
    await coordinator.invalidate("reset")
    infer_release.set()
    await coordinator.wait_for_idle()

    assert infer_calls == [(0, "alpha")]
    assert [m["type"] for m in websocket.messages] == ["session"]
    assert reset_calls == [{"session_id": "session-a", "session_generation": 0}]


@pytest.mark.asyncio
async def test_independent_sessions_keep_buffers_separate(monkeypatch):
    ws_a = DummyWebSocket()
    ws_b = DummyWebSocket()
    session_a = server.InferenceSessionCoordinator(ws_a, session_id="session-a")
    session_b = server.InferenceSessionCoordinator(ws_b, session_id="session-b")
    release = asyncio.Event()
    calls = []

    async def fake_process_inference(session, request_generation, batch, buffer_wait_s):
        calls.append((session.session_id, batch.text, list(batch.annotations)))
        await release.wait()

    monkeypatch.setattr(server, "process_inference", fake_process_inference)

    await session_a.enqueue("chunk-a1", 1.0, "alpha", ["a1"])
    await session_b.enqueue("chunk-b1", 1.0, "beta", ["b1"])
    await asyncio.sleep(0)

    assert ("session-a", "alpha", ["a1"]) in calls
    assert ("session-b", "beta", ["b1"]) in calls

    release.set()
    await asyncio.gather(session_a.wait_for_idle(), session_b.wait_for_idle())


@pytest.mark.asyncio
async def test_seed_goes_ahead_of_the_first_batch_only(monkeypatch):
    coordinator = server.InferenceSessionCoordinator(DummyWebSocket(), session_id="session-a")
    coordinator.seed = "fever for three days"
    batches = []

    async def fake_process_inference(session, request_generation, batch, buffer_wait_s):
        batches.append(batch)

    monkeypatch.setattr(server, "process_inference", fake_process_inference)

    await coordinator.enqueue("chunk-1", 1.0, "then a cough", [])
    await coordinator.wait_for_idle()
    await coordinator.enqueue("chunk-2", 2.0, "and vomiting", [])
    await coordinator.wait_for_idle()

    assert [batch.text for batch in batches] == [
        "fever for three days then a cough",
        "and vomiting",
    ]
    assert [batch.seed for batch in batches] == ["fever for three days", None]


@pytest.mark.asyncio
async def test_reset_drops_the_seed_and_committed_text(monkeypatch):
    async def fake_reset(session_id, generation):
        return None

    monkeypatch.setattr(server, "_reset_inference_session", fake_reset)
    monkeypatch.setattr(server, "storage_enabled", lambda: False)
    coordinator = server.InferenceSessionCoordinator(DummyWebSocket(), session_id="session-a")
    coordinator.seed = "fever for three days"
    coordinator.committed = [("c1", "then a cough")]
    coordinator.buffer.add("then a cough", [], "c1", 1.0)

    await coordinator.invalidate("reset")

    assert coordinator.seed is None
    assert coordinator.committed == []
    assert not coordinator.buffer.has_pending


@pytest.mark.asyncio
async def test_seed_is_sent_again_when_inference_was_unreachable(monkeypatch):
    async def unreachable(path, json):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(server, "inference_client", SimpleNamespace(post=unreachable))
    coordinator = server.InferenceSessionCoordinator(DummyWebSocket(), session_id="session-a")
    batch = server.CoalescedInferenceBatch(
        chunk_id="chunk-1", timestamp=1.0, text="fever for three days then a cough",
        annotations=[], queued_at=0.0, chunk_count=1, seed="fever for three days")

    await server.process_inference(coordinator, 0, batch, buffer_wait_s=0.0)
    assert coordinator.seed == "fever for three days"

    coordinator.seed = None
    await server.process_inference(coordinator, 7, batch, buffer_wait_s=0.0)
    assert coordinator.seed is None


@pytest.mark.asyncio
async def test_restart_infers_on_edited_transcript_then_unshown_text(monkeypatch):
    batches = []

    async def fake_process_inference(session, request_generation, batch, buffer_wait_s):
        batches.append((request_generation, batch.text, batch.annotations,
                        batch.edit_id))

    async def fake_reset(session_id, generation):
        return None

    monkeypatch.setattr(server, "process_inference", fake_process_inference)
    monkeypatch.setattr(server, "_reset_inference_session", fake_reset)
    coordinator = server.InferenceSessionCoordinator(DummyWebSocket(), session_id="session-a")
    coordinator.committed = [("c1", "fever", ["fever"]), ("c2", "for days", []),
                             ("c3", "then cough", ["cough"])]
    coordinator.buffer.add("then cough", [], "c3", 3.0)

    await coordinator.restart("high fever for days", ["high fever"], "c2", "e1")
    await coordinator.wait_for_idle()

    assert batches == [(1, "high fever for days then cough",
                        ["high fever", "cough"], "e1")]
    assert not coordinator.buffer.has_pending
