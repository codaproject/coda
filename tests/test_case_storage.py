"""Tests for per-case storage (coda.app.storage) and its server wiring."""

import asyncio
import io
import json
import logging
import wave
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from coda.app import server
from coda.app.storage import CaseRecorder, case_audio_path, list_cases, load_case
from coda.config import configure_logging, reload_settings

STORAGE_ENV_VARS = (
    "CODA_STORAGE__ENABLED",
    "CODA_STORAGE__BROWSE",
    "CODA_STORAGE__OUTPUT_DIR",
    "CODA_STORAGE__STORE__AUDIO",
    "CODA_STORAGE__STORE__INFERENCE",
    "CODA_LOGGING__FILE",
)


class DummyWebSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, data):
        self.messages.append(data)


class FakeAnnotation:
    def __init__(self, text: str):
        self.text = text

    def to_json(self) -> dict:
        return {"text": self.text}


class GroundedAnnotation:
    """Serializes like a gilda Annotation with one match."""

    def to_json(self) -> dict:
        return {"text": "fever", "start": 0, "end": 5, "matches": [
            {"term": {"db": "MESH", "id": "D005334", "entry_name": "Fever"}}]}


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self) -> dict:
        return dict(self.payload)


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch):
    for var in STORAGE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    reload_settings()
    yield
    monkeypatch.undo()
    reload_settings()


@pytest.fixture
def storage_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CODA_STORAGE__ENABLED", "true")
    monkeypatch.setenv("CODA_STORAGE__OUTPUT_DIR", str(tmp_path))
    reload_settings()
    return tmp_path


@pytest.fixture
def offline_server(monkeypatch):
    async def fake_run_info() -> dict:
        return {"language": "en"}

    async def fake_reset(session_id, generation):
        return None

    monkeypatch.setattr(server, "_run_info", fake_run_info)
    monkeypatch.setattr(server, "_reset_inference_session", fake_reset)
    monkeypatch.setattr(server, "grounder", SimpleNamespace(annotate=lambda text: []))
    monkeypatch.setattr(server, "current_language", "en")


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def case_dirs(root) -> list:
    return sorted(p for p in root.iterdir() if p.is_dir())


def test_recorder_writes_case_files(storage_dir):
    recorder = CaseRecorder("session-abcdef12", 0, {"language": "sw"})
    recorder.write_chunk("c1", 1.5, "fever for three days",
                         [FakeAnnotation("fever")], {"translate_s": 0.1},
                         original_text="homa kwa siku tatu",
                         original_language="sw")
    recorder.write_inference(
        {"chunk_id": "c1", "timestamp": 1.5, "metadata": {"age": 3}},
        {"causes": {"X": {"score": 0.9}}, "questions": ["How old?"],
         "timings": {"request_s": 1.0}},
        shown_at=100.0,
    )
    recorder.close({"age": 3})

    case_dir = recorder.case_dir
    assert case_dir.parent == storage_dir
    assert case_dir.name.endswith("_session-_g0")
    assert (case_dir / "transcript_en.txt").read_text() == "fever for three days\n"
    assert (case_dir / "transcript_sw.txt").read_text() == "homa kwa siku tatu\n"

    [chunk] = read_jsonl(case_dir / "chunks.jsonl")
    assert chunk["annotations"] == [{"text": "fever"}]
    assert chunk["original_language"] == "sw"

    [inference] = read_jsonl(case_dir / "inference.jsonl")
    assert inference["questions"] == ["How old?"]
    assert inference["shown_at"] == 100.0
    assert inference["metadata"] == {"age": 3}

    assert [t["kind"] for t in read_jsonl(case_dir / "timing.jsonl")] == \
        ["chunk", "inference"]

    manifest = json.loads((case_dir / "manifest.json").read_text())
    assert manifest["run_info"] == {"language": "sw"}
    assert manifest["ended_at"] is not None
    assert manifest["metadata"] == {"age": 3}
    assert not (case_dir / "audio.wav").exists()


def test_store_flags_disable_files(storage_dir, monkeypatch):
    monkeypatch.setenv("CODA_STORAGE__STORE__INFERENCE", "false")
    reload_settings()
    recorder = CaseRecorder("session-a", 0, {})
    recorder.write_inference({}, {"questions": ["q"]}, shown_at=1.0)
    recorder.close()
    assert not (recorder.case_dir / "inference.jsonl").exists()


def test_pauses_recorded_in_manifest(storage_dir):
    recorder = CaseRecorder("session-a", 0, {})
    recorder.write_pause()
    recorder.write_resume()
    recorder.write_pause()
    recorder.close()

    manifest = json.loads((recorder.case_dir / "manifest.json").read_text())
    first, second = manifest["pauses"]
    assert first["paused_at"] <= first["resumed_at"]
    assert second["resumed_at"] is None


def test_audio_is_written_as_wav(storage_dir, monkeypatch):
    monkeypatch.setenv("CODA_STORAGE__STORE__AUDIO", "true")
    reload_settings()
    recorder = CaseRecorder("session-a", 0, {})
    recorder.write_audio(b"\x00\x01" * 1600)
    recorder.close()
    with wave.open(str(recorder.case_dir / "audio.wav")) as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 1600


@pytest.mark.asyncio
async def test_nothing_written_when_storage_disabled(offline_server, tmp_path,
                                                     monkeypatch):
    monkeypatch.setenv("CODA_STORAGE__OUTPUT_DIR", str(tmp_path))
    reload_settings()
    session = server.InferenceSessionCoordinator(DummyWebSocket(), "session-a")
    await session.start_case()
    event = SimpleNamespace(id="c1", timestamp=0.0, text="fever")
    await server._handle_committed(session, event, direct_translate=False)
    await session.invalidate("reset")
    await session.invalidate("disconnect")

    assert session.recorder is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_reset_starts_new_case(offline_server, storage_dir):
    session = server.InferenceSessionCoordinator(DummyWebSocket(), "session-a")
    await session.start_case()
    event = SimpleNamespace(id="c1", timestamp=0.0, text="fever")
    await server._handle_committed(session, event, direct_translate=False)
    await session.invalidate("reset")
    await session.invalidate("disconnect")

    assert session.recorder is None
    first, second = case_dirs(storage_dir)
    assert first.name.endswith("_g0") and second.name.endswith("_g1")
    assert len(read_jsonl(first / "chunks.jsonl")) == 1
    assert (second / "chunks.jsonl").read_text() == ""
    for case_dir in (first, second):
        manifest = json.loads((case_dir / "manifest.json").read_text())
        assert manifest["ended_at"] is not None


@pytest.mark.asyncio
async def test_targeted_reset_only_rotates_target_case(offline_server,
                                                       storage_dir):
    session_a = server.InferenceSessionCoordinator(DummyWebSocket(), "session-a")
    session_b = server.InferenceSessionCoordinator(DummyWebSocket(), "session-b")
    await session_a.start_case()
    await session_b.start_case()
    recorder_b = session_b.recorder
    original_sessions = set(server.active_inference_sessions)
    server.active_inference_sessions.clear()
    server.active_inference_sessions.update({session_a, session_b})
    try:
        await server.reset_session(server.ResetRequest(session_id="session-a"))
    finally:
        server.active_inference_sessions.clear()
        server.active_inference_sessions.update(original_sessions)

    assert session_a.recorder.case_id.endswith("_g1")
    assert session_b.recorder is recorder_b
    session_a.end_case()
    session_b.end_case()


@pytest.mark.asyncio
async def test_inference_results_recorded_unless_stale(offline_server,
                                                       storage_dir, monkeypatch):
    session = server.InferenceSessionCoordinator(DummyWebSocket(), "session-a")
    await session.start_case()

    async def fake_post(path, json):
        return FakeResponse({"chunk_id": json["chunk_id"], "questions": ["q1"]})

    monkeypatch.setattr(server, "inference_client",
                        SimpleNamespace(post=fake_post))
    batch = server.CoalescedInferenceBatch(
        chunk_id="c1", timestamp=0.0, text="fever", annotations=[],
        queued_at=0.0, chunk_count=1)
    await server.process_inference(session, 0, batch, buffer_wait_s=0.0)
    await server.process_inference(session, 7, batch, buffer_wait_s=0.0)
    case_dir = session.recorder.case_dir
    session.end_case()

    [record] = read_jsonl(case_dir / "inference.jsonl")
    assert record["questions"] == ["q1"]
    assert record["shown_at"] is not None


def test_configure_logging_writes_to_file(monkeypatch, tmp_path):
    log_file = tmp_path / "logs" / "coda.log"
    monkeypatch.setenv("CODA_LOGGING__FILE", str(log_file))
    reload_settings()
    root = logging.getLogger()
    previous_handlers, previous_level = root.handlers[:], root.level
    try:
        configure_logging()
        logging.getLogger("coda.test").info("hello storage")
        for handler in root.handlers:
            handler.flush()
    finally:
        for handler in root.handlers:
            if handler not in previous_handlers:
                handler.close()
        root.handlers[:] = previous_handlers
        root.setLevel(previous_level)
    assert "hello storage" in log_file.read_text()


MALARIA = {"name": "Malaria", "identifiers": {"icd10": "B54"}, "score": 0.7}
SEPSIS = {"name": "Sepsis", "identifiers": {"icd10": "A41.9"}, "score": 0.2}
PROFILE = {"profile": {"age": {"value": 3, "unit": "years"},
                       "stillbirth": False}}


def record_case(session_id: str, close: bool = True) -> CaseRecorder:
    recorder = CaseRecorder(session_id, 0, {"language": "sw"})
    recorder.write_chunk("c1", 1.5, "fever for three days",
                         [GroundedAnnotation()], {},
                         original_text="homa kwa siku tatu",
                         original_language="sw")
    recorder.write_inference(
        {"chunk_id": "c1", "metadata": PROFILE},
        {"causes": {"icd10:B54": MALARIA, "icd10:A41.9": SEPSIS},
         "reasoning": "Fever in a malaria area.", "questions": ["How old?"]},
        shown_at=100.0,
    )
    if close:
        recorder.close(PROFILE)
    return recorder


def test_load_case_reads_back_recorded_case(storage_dir):
    recorder = record_case("aaaa0001")

    case = load_case(recorder.case_id)
    assert case["case_id"] == recorder.case_id
    assert case["ended_at"] is not None
    assert case["top_cause"] == MALARIA
    assert case["metadata"] == PROFILE
    [chunk] = case["chunks"]
    assert chunk["transcript"] == "fever for three days"
    assert chunk["original_transcript"] == "homa kwa siku tatu"
    assert chunk["annotations"] == [{"text": "fever", "start": 0, "end": 5,
                                     "curie": "mesh:D005334", "name": "Fever"}]
    assert case["transcripts"] == {"en": "fever for three days\n",
                                   "sw": "homa kwa siku tatu\n"}
    [inference] = case["inference"]
    assert inference["questions"] == ["How old?"]


def test_list_cases_skips_empty_cases_newest_first(storage_dir):
    older = record_case("aaaa0001")
    CaseRecorder("bbbb0002", 0, {}).close()
    newer = record_case("cccc0003")

    cases = list_cases()
    assert [c["case_id"] for c in cases] == [newer.case_id, older.case_id]
    assert cases[0]["chunk_count"] == 1
    assert cases[0]["inference_count"] == 1
    assert cases[0]["language"] == "sw"
    started = datetime.fromisoformat(cases[0]["started_at"])
    assert started.utcoffset() is not None
    assert started == datetime.fromisoformat(newer.manifest["started_at"]).astimezone()


def test_audio_without_frames_counts_as_none(storage_dir, monkeypatch):
    monkeypatch.setenv("CODA_STORAGE__STORE__AUDIO", "true")
    reload_settings()
    silent = record_case("aaaa0001")
    spoken = CaseRecorder("bbbb0002", 0, {})
    spoken.write_audio(b"\x00\x01" * 1600)
    spoken.close()

    assert load_case(silent.case_id)["has_audio"] is False
    assert case_audio_path(silent.case_id) is None
    assert case_audio_path(spoken.case_id) == spoken.case_dir / "audio.wav"


def test_unclosed_case_keeps_what_was_written(storage_dir):
    recorder = record_case("aaaa0001", close=False)
    with open(recorder.case_dir / "chunks.jsonl", "a") as f:
        f.write('{"chunk_id": "c2", "text": "cut sh')

    case = load_case(recorder.case_id)
    recorder.close()
    assert case["ended_at"] is None
    assert [c["chunk_id"] for c in case["chunks"]] == ["c1"]
    assert case["metadata"] == PROFILE


def test_items_not_stored_load_as_none(storage_dir, monkeypatch):
    monkeypatch.setenv("CODA_STORAGE__STORE__INFERENCE", "false")
    reload_settings()
    recorder = record_case("aaaa0001")

    case = load_case(recorder.case_id)
    assert case["inference"] is None
    assert case["top_cause"] is None
    assert len(case["chunks"]) == 1


def test_unknown_or_unsafe_case_ids_are_rejected(storage_dir):
    recorder = record_case("aaaa0001")
    assert load_case("../" + recorder.case_id) is None
    assert load_case("no-such-case") is None
    assert case_audio_path(recorder.case_id) is None


def test_case_endpoints_are_off_unless_browsable(storage_dir):
    record_case("aaaa0001")
    client = TestClient(server.app)
    assert client.get("/cases").status_code == 404
    assert client.get("/settings").json()["cases_browsable"] is False


def test_case_endpoints_serve_stored_cases(storage_dir, monkeypatch):
    monkeypatch.setenv("CODA_STORAGE__BROWSE", "true")
    monkeypatch.setenv("CODA_STORAGE__STORE__AUDIO", "true")
    reload_settings()
    closed = CaseRecorder("aaaa0001", 0, {})
    closed.write_audio(b"\x00\x01" * 1600)
    closed.write_chunk("c1", 0.0, "fever for three days", [], {})
    closed.close()
    recording = CaseRecorder("bbbb0002", 0, {})
    recording.write_chunk("c1", 0.0, "she had a cough", [], {})
    monkeypatch.setattr(server, "active_inference_sessions",
                        [SimpleNamespace(recorder=recording)])
    client = TestClient(server.app)

    cases = {c["case_id"]: c for c in client.get("/cases").json()}
    assert cases[closed.case_id]["in_progress"] is False
    assert cases[recording.case_id]["in_progress"] is True

    case = client.get(f"/cases/{closed.case_id}").json()
    assert case["chunks"][0]["transcript"] == "fever for three days"
    assert client.get("/cases/no-such-case").status_code == 404

    audio = client.get(f"/cases/{closed.case_id}/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    assert client.get(f"/cases/{recording.case_id}/audio").status_code == 404
    recording.close()


def test_settings_report_storage_methods(storage_dir, monkeypatch):
    client = TestClient(server.app)
    reported = client.get("/settings").json()
    assert reported["storage_enabled"] is True
    assert reported["cases_browsable"] is False
    assert reported["storage_audio"] is False

    monkeypatch.setenv("CODA_STORAGE__BROWSE", "true")
    monkeypatch.setenv("CODA_STORAGE__STORE__AUDIO", "true")
    reload_settings()
    reported = client.get("/settings").json()
    assert reported["cases_browsable"] is True
    assert reported["storage_audio"] is True

    monkeypatch.setenv("CODA_STORAGE__ENABLED", "false")
    reload_settings()
    assert client.get("/settings").json()["storage_audio"] is False


class ClosingWebSocket(DummyWebSocket):
    """Accepts, then disconnects on the first receive."""

    async def accept(self):
        pass

    async def receive(self) -> dict:
        return {"type": "websocket.disconnect", "code": 1000}


@pytest.fixture
def idle_transcription(offline_server, monkeypatch):
    async def wait_forever(websocket, queue, inference_session):
        await asyncio.Event().wait()

    monkeypatch.setattr(server, "consume_transcripts", wait_forever)


async def test_session_message_carries_case_id(storage_dir, idle_transcription):
    websocket = ClosingWebSocket()
    await server.websocket_endpoint(websocket)

    [case_dir] = case_dirs(storage_dir)
    assert websocket.messages[0]["type"] == "session"
    assert websocket.messages[0]["case_id"] == case_dir.name
    manifest = json.loads((case_dir / "manifest.json").read_text())
    assert manifest["ended_at"] is not None


async def test_session_message_has_no_case_id_without_storage(idle_transcription):
    websocket = ClosingWebSocket()
    await server.websocket_endpoint(websocket)
    assert websocket.messages[0]["case_id"] is None


async def test_reset_tells_the_client_its_new_case(storage_dir, offline_server):
    websocket = DummyWebSocket()
    session = server.InferenceSessionCoordinator(websocket, "session-a")
    await session.start_case()
    first_case = session.recorder.case_id

    await session.invalidate("reset")

    assert websocket.messages[-1] == {
        "type": "session",
        "session_id": "session-a",
        "session_generation": 1,
        "case_id": session.recorder.case_id,
    }
    assert session.recorder.case_id != first_case
    session.end_case()


def edit_manifest(case_dir, **changes):
    path = case_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(changes)
    path.write_text(json.dumps(manifest))
    return manifest


def test_reopened_case_appends_a_segment(storage_dir):
    first = record_case("aaaa0001")
    second = CaseRecorder("bbbb0002", 0, {"language": "en"}, case_id=first.case_id)
    second.write_audio(b"\x00\x01")
    second.write_chunk("c2", 2.0, "then a cough", [], {})
    second.write_inference({"chunk_id": "c2"}, {"causes": {"icd10:B54": MALARIA}},
                           shown_at=200.0)
    second.close(PROFILE)

    case_dir = first.case_dir
    assert second.case_dir == case_dir
    assert second.segment == 1
    chunks = read_jsonl(case_dir / "chunks.jsonl")
    assert [(c["chunk_id"], c["segment"]) for c in chunks] == [("c1", 0), ("c2", 1)]
    assert (case_dir / "transcript_en.txt").read_text() == \
        "fever for three days\nthen a cough\n"
    assert len(read_jsonl(case_dir / "inference.jsonl")) == 2
    assert [t["kind"] for t in read_jsonl(case_dir / "timing.jsonl")] == \
        ["chunk", "inference", "chunk", "inference"]

    manifest = json.loads((case_dir / "manifest.json").read_text())
    first_segment, second_segment = manifest["segments"]
    assert first_segment["session_id"] == "aaaa0001"
    assert second_segment["session_id"] == "bbbb0002"
    assert second_segment["run_info"] == {"language": "en"}
    assert manifest["run_info"] == {"language": "sw"}
    assert manifest["pauses"] == [{"paused_at": first_segment["ended_at"],
                                   "resumed_at": second_segment["started_at"],
                                   "gap": True}]
    assert manifest["ended_at"] == second_segment["ended_at"]
    assert [p["part"] for p in load_case(first.case_id)["segments"]] == [0, 1]


def test_duration_leaves_out_the_time_between_segments(storage_dir):
    first = record_case("aaaa0001")
    started = datetime.now() - timedelta(hours=1)
    ended = (started + timedelta(seconds=30)).isoformat()
    segment = json.loads((first.case_dir / "manifest.json").read_text())["segments"][0]
    edit_manifest(first.case_dir, started_at=started.isoformat(), ended_at=ended,
                  segments=[{**segment, "started_at": started.isoformat(),
                             "ended_at": ended}])
    second = CaseRecorder("bbbb0002", 0, {}, case_id=first.case_id)
    second.write_audio(b"\x00\x01")
    second.close()

    assert load_case(first.case_id)["duration_s"] == 30


def test_case_recorded_before_segments_reopens_as_its_second(storage_dir):
    first = record_case("aaaa0001")
    manifest = json.loads((first.case_dir / "manifest.json").read_text())
    del manifest["segments"]
    (first.case_dir / "manifest.json").write_text(json.dumps(manifest))
    assert [p["part"] for p in load_case(first.case_id)["segments"]] == [0]

    second = CaseRecorder("bbbb0002", 0, {}, case_id=first.case_id)
    second.write_audio(b"\x00\x01")
    second.close()

    reopened = json.loads((first.case_dir / "manifest.json").read_text())
    assert [s["session_id"] for s in reopened["segments"]] == ["aaaa0001", "bbbb0002"]
    assert reopened["segments"][0]["ended_at"] == manifest["ended_at"]


def test_reopening_a_crashed_case_closes_it_at_its_last_write(storage_dir):
    crashed = record_case("aaaa0001")
    paused_at = datetime.now().isoformat()
    segment = json.loads((crashed.case_dir / "manifest.json").read_text())["segments"][0]
    edit_manifest(crashed.case_dir, ended_at=None,
                  pauses=[{"paused_at": paused_at, "resumed_at": None}],
                  segments=[{**segment, "ended_at": None}])
    with open(crashed.case_dir / "chunks.jsonl", "a") as f:
        f.write('{"chunk_id": "c2", "text": "cut sh')
    last_write = datetime.fromtimestamp(
        max(p.stat().st_mtime for p in crashed.case_dir.iterdir())).isoformat()

    reopened = CaseRecorder("bbbb0002", 0, {}, case_id=crashed.case_id)
    reopened.write_audio(b"\x00\x01")
    reopened.write_chunk("c3", 3.0, "she had a cough", [], {})
    reopened.close()

    manifest = json.loads((crashed.case_dir / "manifest.json").read_text())
    assert manifest["segments"][0]["ended_at"] == last_write
    pause, gap = manifest["pauses"]
    assert pause == {"paused_at": paused_at, "resumed_at": last_write}
    assert gap["paused_at"] == last_write
    assert [c["chunk_id"] for c in load_case(crashed.case_id)["chunks"]] == ["c1", "c3"]


def test_segment_that_received_no_audio_is_dropped(storage_dir, monkeypatch):
    monkeypatch.setenv("CODA_STORAGE__STORE__AUDIO", "true")
    reload_settings()
    first = record_case("aaaa0001")
    before = json.loads((first.case_dir / "manifest.json").read_text())

    CaseRecorder("bbbb0002", 0, {}, case_id=first.case_id).close()

    after = json.loads((first.case_dir / "manifest.json").read_text())
    assert after["segments"] == before["segments"]
    assert after["pauses"] == before["pauses"] == []
    assert after["ended_at"] == before["ended_at"]
    assert not (first.case_dir / "audio_1.wav").exists()


def test_case_audio_is_kept_and_served_per_part(storage_dir, monkeypatch):
    monkeypatch.setenv("CODA_STORAGE__BROWSE", "true")
    monkeypatch.setenv("CODA_STORAGE__STORE__AUDIO", "true")
    reload_settings()
    first = CaseRecorder("aaaa0001", 0, {})
    first.write_audio(b"\x00\x01" * 1600)
    first.write_chunk("c1", 0.0, "fever for three days", [], {})
    first.close()
    second = CaseRecorder("bbbb0002", 0, {}, case_id=first.case_id)
    second.write_audio(b"\x00\x01" * 3200)
    second.close()
    client = TestClient(server.app)

    case = client.get(f"/cases/{first.case_id}").json()
    assert [p["audio_s"] for p in case["segments"]] == [0.1, 0.2]
    assert case_audio_path(first.case_id) == first.case_dir / "audio.wav"
    assert case_audio_path(first.case_id, 1) == first.case_dir / "audio_1.wav"
    response = client.get(f"/cases/{first.case_id}/audio?part=1")
    assert response.status_code == 200
    with wave.open(io.BytesIO(response.content)) as wav:
        assert wav.getnframes() == 3200
    assert client.get(f"/cases/{first.case_id}/audio?part=2").status_code == 404


class RefusedWebSocket(ClosingWebSocket):
    def __init__(self):
        super().__init__()
        self.close_code = None

    async def close(self, code: int = 1000):
        self.close_code = code


class ScriptedWebSocket(DummyWebSocket):
    def __init__(self, messages: list[dict]):
        super().__init__()
        self.incoming = list(messages)

    async def receive(self) -> dict:
        return self.incoming.pop(0)


async def test_connection_continues_a_stored_case(storage_dir, offline_server,
                                                  monkeypatch):
    async def hear_audio(websocket, queue, inference_session):
        inference_session.recorder.write_audio(b"\x00\x01")
        await asyncio.Event().wait()

    monkeypatch.setattr(server, "consume_transcripts", hear_audio)
    first = record_case("aaaa0001")
    websocket = ClosingWebSocket()
    await server.websocket_endpoint(websocket, case_id=first.case_id)

    assert case_dirs(storage_dir) == [first.case_dir]
    assert websocket.messages[0]["case_id"] == first.case_id
    manifest = json.loads((first.case_dir / "manifest.json").read_text())
    assert len(manifest["segments"]) == 2
    assert manifest["ended_at"] is not None


async def test_continuing_an_unknown_case_starts_a_new_one(storage_dir,
                                                           idle_transcription):
    websocket = ClosingWebSocket()
    await server.websocket_endpoint(websocket, case_id="no-such-case")

    [case_dir] = case_dirs(storage_dir)
    assert websocket.messages[0]["case_id"] == case_dir.name


async def test_case_being_recorded_is_not_continued(storage_dir, idle_transcription,
                                                    monkeypatch):
    monkeypatch.setattr(server, "CASE_RELEASE_WAIT_S", 0.2)
    live = server.InferenceSessionCoordinator(DummyWebSocket(), "aaaa0001")
    await live.start_case()
    monkeypatch.setattr(server, "active_inference_sessions", {live})

    websocket = RefusedWebSocket()
    await server.websocket_endpoint(websocket, case_id=live.recorder.case_id)

    assert websocket.close_code == server.CASE_IN_USE_CLOSE_CODE
    [error] = websocket.messages
    assert error["type"] == "error" and error["fatal"] is True
    assert server.active_inference_sessions == {live}
    live.end_case()


async def test_case_released_while_waiting_is_continued(storage_dir,
                                                        idle_transcription,
                                                        monkeypatch):
    live = server.InferenceSessionCoordinator(DummyWebSocket(), "aaaa0001")
    await live.start_case()
    case_id = live.recorder.case_id
    monkeypatch.setattr(server, "active_inference_sessions", {live})

    async def stop_shortly():
        await asyncio.sleep(0.3)
        await live.invalidate("disconnect")

    stopping = asyncio.create_task(stop_shortly())
    websocket = ClosingWebSocket()
    await server.websocket_endpoint(websocket, case_id=case_id)
    await stopping

    assert websocket.messages[0]["case_id"] == case_id
    assert case_dirs(storage_dir) == [storage_dir / case_id]


async def test_disconnect_releases_the_case_before_resetting_inference(
        storage_dir, offline_server, monkeypatch):
    session = server.InferenceSessionCoordinator(DummyWebSocket(), "aaaa0001")
    await session.start_case()
    monkeypatch.setattr(server, "active_inference_sessions", {session})
    recording_during_reset = []

    async def fake_reset(session_id, generation):
        recording_during_reset.append(server._recording_case_ids())

    monkeypatch.setattr(server, "_reset_inference_session", fake_reset)
    await session.invalidate("disconnect")

    assert recording_during_reset == [set()]


async def test_seed_is_kept_by_a_new_case_only(storage_dir, offline_server):
    seed = {"type": "websocket.receive",
            "text": json.dumps({"type": "seed", "text": "fever for three days"})}
    disconnect = {"type": "websocket.disconnect", "code": 1000}

    new = server.InferenceSessionCoordinator(DummyWebSocket(), "aaaa0001")
    await new.start_case()
    with pytest.raises(WebSocketDisconnect):
        await server.capture_audio(ScriptedWebSocket([seed, disconnect]),
                                   asyncio.Queue(), new)
    case_id = new.recorder.case_id
    new.end_case()

    reopened = server.InferenceSessionCoordinator(DummyWebSocket(), "bbbb0002")
    assert await reopened.continue_case(case_id)
    with pytest.raises(WebSocketDisconnect):
        await server.capture_audio(ScriptedWebSocket([seed, disconnect]),
                                   asyncio.Queue(), reopened)
    reopened.end_case()

    assert new.seed == reopened.seed == "fever for three days"
    [chunk] = load_case(case_id)["chunks"]
    assert chunk["imported"] is True
    assert chunk["transcript"] == "fever for three days"
