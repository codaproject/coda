import asyncio
import json

import pytest
from fastapi import WebSocketDisconnect

from coda.app import server
from coda.dialogue import StreamingTranscriber, TranscriptEvent


class FakeReceiveWebSocket:
    def __init__(self, messages):
        self.messages = list(messages)

    async def receive(self):
        return self.messages.pop(0)


class FakeSendWebSocket:
    def __init__(self, log):
        self.log = log

    async def send_json(self, data):
        self.log.append(data["type"])


class LaggingTranscriber(StreamingTranscriber):
    """Commits each frame's text only once the next frame arrives, like a
    streaming backend holding its last words until more audio comes."""

    def __init__(self, log):
        self.log = log

    async def stream(self, audio, *, language="en", task="transcribe"):
        held = None
        async for data in audio:
            if data == bytes(server.PAUSE_SILENCE_BYTES):
                self.log.append("silence")
            if held:
                yield TranscriptEvent(id=held, timestamp=0.0, text=held,
                                      committed=True)
            held = None if not data.strip(b"\0") else data.decode()
        if held:
            yield TranscriptEvent(id=held, timestamp=0.0, text=held,
                                  committed=True)


@pytest.mark.asyncio
async def test_capture_applies_control_messages_on_arrival():
    websocket = FakeReceiveWebSocket([
        {"type": "websocket.receive", "bytes": b"a"},
        {"type": "websocket.receive", "text": json.dumps({"type": "pause"})},
        {"type": "websocket.receive", "text": "not json"},
        {"type": "websocket.receive", "text": json.dumps({"type": "stop"})},
        {"type": "websocket.disconnect", "code": 1000},
    ])
    session = server.InferenceSessionCoordinator(None, session_id="s")
    queue = asyncio.Queue()
    with pytest.raises(WebSocketDisconnect):
        await server.capture_audio(websocket, queue, session)
    assert session.paused
    assert [queue.get_nowait() for _ in range(queue.qsize())] == \
        [b"a", "pause", None, None]


@pytest.mark.asyncio
async def test_no_inference_while_paused(monkeypatch):
    log = []

    async def fake_handle_committed(session, event, direct_translate):
        log.append(f"commit {event.text}")
        return event.id, event.timestamp, event.text, []

    async def fake_start_inference(session, chunk_id, timestamp, text, anns):
        log.append(f"infer {text}")

    async def wait_for(entry):
        while entry not in log:
            await asyncio.sleep(0.05)

    monkeypatch.setattr(server, "transcriber", LaggingTranscriber(log))
    monkeypatch.setattr(server, "_handle_committed", fake_handle_committed)
    monkeypatch.setattr(server, "_start_inference", fake_start_inference)
    monkeypatch.setattr(server, "INFERENCE_MIN_WORDS", 2)

    session = server.InferenceSessionCoordinator(None, session_id="s")
    queue = asyncio.Queue()
    consume = asyncio.create_task(
        server.consume_transcripts(FakeSendWebSocket(log), queue, session))
    # Paused before the transcriber has caught up with the earlier audio
    queue.put_nowait(b"fever for days")
    session.paused = True
    queue.put_nowait("pause")
    await asyncio.wait_for(wait_for("commit fever for days"), 2)
    await asyncio.sleep(1.5)
    assert "infer fever for days" not in log

    session.paused = False
    await asyncio.wait_for(wait_for("infer fever for days"), 3)
    queue.put_nowait(b"then cough")
    queue.put_nowait(None)
    await asyncio.wait_for(consume, 2)

    assert log == ["silence", "commit fever for days", "infer fever for days",
                   "commit then cough", "infer then cough", "stopped"]


@pytest.mark.asyncio
async def test_stopped_is_sent_after_the_last_inference(monkeypatch):
    log = []

    async def fake_handle_committed(session, event, direct_translate):
        return event.id, event.timestamp, event.text, []

    async def slow_process_inference(session, request_generation, batch,
                                     buffer_wait_s):
        await asyncio.sleep(0.2)
        log.append(f"infer {batch.text}")

    monkeypatch.setattr(server, "transcriber", LaggingTranscriber(log))
    monkeypatch.setattr(server, "_handle_committed", fake_handle_committed)
    monkeypatch.setattr(server, "process_inference", slow_process_inference)
    monkeypatch.setattr(server, "INFERENCE_MIN_WORDS", 100)

    session = server.InferenceSessionCoordinator(None, session_id="s")
    queue = asyncio.Queue()
    queue.put_nowait(b"fever for days")
    queue.put_nowait(None)
    await asyncio.wait_for(server.consume_transcripts(
        FakeSendWebSocket(log), queue, session), 2)

    assert log == ["infer fever for days", "stopped"]


@pytest.mark.asyncio
async def test_capture_takes_one_seed_before_any_audio():
    def seed(text):
        return {"type": "websocket.receive",
                "text": json.dumps({"type": "seed", "text": text})}

    disconnect = {"type": "websocket.disconnect", "code": 1000}
    session = server.InferenceSessionCoordinator(None, session_id="s")
    with pytest.raises(WebSocketDisconnect):
        await server.capture_audio(
            FakeReceiveWebSocket([seed("fever for days"), seed("again"), disconnect]),
            asyncio.Queue(), session)
    assert session.seed == "fever for days"

    late = server.InferenceSessionCoordinator(None, session_id="s")
    with pytest.raises(WebSocketDisconnect):
        await server.capture_audio(
            FakeReceiveWebSocket([{"type": "websocket.receive", "bytes": b"a"},
                                  seed("fever for days"), disconnect]),
            asyncio.Queue(), late)
    assert late.seed is None
