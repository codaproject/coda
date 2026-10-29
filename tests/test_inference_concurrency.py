import asyncio
import threading
import time

import httpx
import pytest

from coda.inference.agent import InferenceServer
from coda.inference.champs_finetuned.agent import ChampsFinetunedInferenceAgent
from coda.inference.champs_prompted.icd10_agent import ChampsPromptedIcd10Agent
from coda.inference.champs_prompted.base_agent import render_profile
from coda.metadata import Metadata


class FakeLLMClient:
    def __init__(self, *, block_event=None, entered_event=None,
                 fail=False, active_counter=None):
        self.block_event = block_event
        self.entered_event = entered_event
        self.fail = fail
        self.active_counter = active_counter
        self.thread_ids = []
        self.user_prompts = []

    def call_with_schema(self, **kwargs):
        self.thread_ids.append(threading.get_ident())
        self.user_prompts.append(kwargs["user_prompt"])
        counter = self.active_counter
        if counter is not None:
            with counter["lock"]:
                counter["active"] += 1
                counter["max_active"] = max(counter["max_active"], counter["active"])
        try:
            if self.entered_event is not None:
                self.entered_event.set()
            if self.block_event is not None:
                self.block_event.wait(1.0)
            if self.fail:
                raise RuntimeError("boom")
            return {
                "reasoning": "test reasoning",
                "top_causes": [
                    {"cause_name": "Anemias", "probability": 1.0}
                ],
                "questions": ["q1", "q2", "q3"],
            }
        finally:
            if counter is not None:
                with counter["lock"]:
                    counter["active"] -= 1


async def wait_for_thread_event(event: threading.Event, timeout: float = 1.0):
    deadline = time.monotonic() + timeout
    while not event.is_set():
        if time.monotonic() >= deadline:
            raise TimeoutError("thread event was not set in time")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_champs_llm_runs_off_event_loop_thread():
    client = FakeLLMClient()
    agent = ChampsPromptedIcd10Agent(client)
    loop_thread = threading.get_ident()

    result = await agent.process_chunk("chunk-1", "Patient had fever.", [])

    assert result["causes"]
    assert client.thread_ids
    assert client.thread_ids[0] != loop_thread


@pytest.mark.asyncio
async def test_champs_llm_prompt_tags_clinical_notes():
    client = FakeLLMClient()
    agent = ChampsPromptedIcd10Agent(client)
    notes = [{"filename": "labs.txt", "text": "Hb 5"},
             {"filename": "discharge.txt", "text": "Sepsis"}]

    await agent.set_clinical_notes("note-1", notes, "Child had fever.", [])

    prompt = client.user_prompts[-1]
    assert "Child had fever." in prompt
    assert '<clinical_note filename="labs.txt">\nHb 5\n</clinical_note>\n' \
        '<clinical_note filename="discharge.txt">\nSepsis\n' in prompt


@pytest.mark.asyncio
async def test_champs_llm_prompt_includes_profile():
    client = FakeLLMClient()
    agent = ChampsPromptedIcd10Agent(client)
    metadata = {"profile": {"age": {"value": 18, "unit": "months"},
                            "sex": "male", "date_of_death": "2026-03-14",
                            "manner_of_death": "self_harm"}}

    await agent.process_chunk("c1", "Child had fever.", [], metadata=metadata)

    assert "## INPUT\n- age at death: 18 months\n- sex: male\n" \
        "- date of death: 2026-03-14\n- manner of death: self harm\n" \
        "- narrative:" in client.user_prompts[-1]


@pytest.mark.asyncio
async def test_champs_llm_exception_returns_safe_failure():
    client = FakeLLMClient(fail=True)
    agent = ChampsPromptedIcd10Agent(client)

    result = await agent.infer("chunk-1", "Patient had fever.", [])

    assert result == {
        "causes": {},
        "reasoning": "LLM API call raised an exception.",
        "questions": [],
    }


@pytest.mark.asyncio
async def test_llm_concurrency_bound_is_respected():
    counter = {"active": 0, "max_active": 0, "lock": threading.Lock()}

    class SlowLLMClient:
        def call_with_schema(self, **kwargs):
            with counter["lock"]:
                counter["active"] += 1
                counter["max_active"] = max(counter["max_active"], counter["active"])
            try:
                time.sleep(0.2)
                return {
                    "reasoning": "test reasoning",
                    "top_causes": [
                        {"cause_name": "Anemias", "probability": 1.0}
                    ],
                    "questions": ["q1", "q2", "q3"],
                }
            finally:
                with counter["lock"]:
                    counter["active"] -= 1

    prototype = ChampsPromptedIcd10Agent(SlowLLMClient())
    agent_a = prototype.create_session_agent()
    agent_b = prototype.create_session_agent()

    await asyncio.gather(
        agent_a.process_chunk("chunk-a", "one", []),
        agent_b.process_chunk("chunk-b", "two", []),
    )

    assert counter["max_active"] == 1


@pytest.mark.asyncio
async def test_health_endpoint_remains_responsive_while_llm_is_blocked():
    entered = threading.Event()
    release = threading.Event()
    agent = ChampsPromptedIcd10Agent(FakeLLMClient(
        block_event=release,
        entered_event=entered,
    ))
    server = InferenceServer(agent, host="127.0.0.1", port=5123)

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as infer_client:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as health_client:
            infer_task = asyncio.create_task(infer_client.post("/infer", json={
                "chunk_id": "chunk-1",
                "text": "Patient had fever.",
                "annotations": [],
                "session_id": "session-a",
                "session_generation": 0,
            }))
            try:
                await wait_for_thread_event(entered, timeout=1.0)
                health_response = await asyncio.wait_for(
                    health_client.get("/health"),
                    timeout=0.2,
                )
                assert health_response.status_code == 200
                assert health_response.json() == {"status": "healthy"}
            finally:
                release.set()

            infer_response = await infer_task

    assert infer_response.status_code == 200


def test_inference_server_keeps_sessions_isolated():
    from fastapi.testclient import TestClient
    server = InferenceServer(ChampsPromptedIcd10Agent(FakeLLMClient()))
    client = TestClient(server.app)

    resp_a1 = client.post("/infer", json={
        "chunk_id": "a-1",
        "text": "alpha",
        "annotations": [],
        "session_id": "session-a",
        "session_generation": 0,
    })
    resp_b1 = client.post("/infer", json={
        "chunk_id": "b-1",
        "text": "beta",
        "annotations": [],
        "session_id": "session-b",
        "session_generation": 0,
    })
    resp_a2 = client.post("/infer", json={
        "chunk_id": "a-2",
        "text": "gamma",
        "annotations": [],
        "session_id": "session-a",
        "session_generation": 0,
    })

    assert resp_a1.json()["chunks_processed"] == 1
    assert resp_b1.json()["chunks_processed"] == 1
    assert resp_a2.json()["chunks_processed"] == 2


def test_inference_server_reset_clears_only_target_session():
    from fastapi.testclient import TestClient
    server = InferenceServer(ChampsPromptedIcd10Agent(FakeLLMClient()))
    client = TestClient(server.app)

    client.post("/infer", json={
        "chunk_id": "a-1",
        "text": "alpha",
        "annotations": [],
        "session_id": "session-a",
        "session_generation": 0,
    })
    client.post("/infer", json={
        "chunk_id": "b-1",
        "text": "beta",
        "annotations": [],
        "session_id": "session-b",
        "session_generation": 0,
    })

    reset_resp = client.post("/reset", json={
        "session_id": "session-a",
        "session_generation": 0,
    })
    resp_a_new = client.post("/infer", json={
        "chunk_id": "a-2",
        "text": "gamma",
        "annotations": [],
        "session_id": "session-a",
        "session_generation": 1,
    })
    resp_b2 = client.post("/infer", json={
        "chunk_id": "b-2",
        "text": "delta",
        "annotations": [],
        "session_id": "session-b",
        "session_generation": 0,
    })

    assert reset_resp.status_code == 200
    assert resp_a_new.json()["chunks_processed"] == 1
    assert resp_b2.json()["chunks_processed"] == 2


FINETUNED_MENU = {"C1": "Cause one", "C2": "Cause two"}


class FakeMedGemmaModel:
    def __init__(self, counter=None):
        self.counter = counter

    def score_candidates(self, messages, candidates, batch_size=8):
        counter = self.counter
        if counter is not None:
            with counter["lock"]:
                counter["active"] += 1
                counter["max_active"] = max(counter["max_active"], counter["active"])
        try:
            time.sleep(0.1)
            return [0.9, 0.1][: len(candidates)]
        finally:
            if counter is not None:
                with counter["lock"]:
                    counter["active"] -= 1


def test_finetuned_create_session_agent_shares_model_and_semaphore():
    model = FakeMedGemmaModel()
    agent = ChampsFinetunedInferenceAgent(
        menu=FINETUNED_MENU, adapter_path="x", model=model)
    session = agent.create_session_agent()

    assert session._model is model
    assert session.llm_semaphore is agent.llm_semaphore
    assert session.menu == FINETUNED_MENU


@pytest.mark.asyncio
async def test_finetuned_infer_returns_causes():
    agent = ChampsFinetunedInferenceAgent(
        menu=FINETUNED_MENU, adapter_path="x", model=FakeMedGemmaModel())
    result = await agent.process_chunk("chunk-1", "Patient had fever.", [])

    assert result["causes"]


@pytest.mark.asyncio
async def test_finetuned_concurrency_bound_shared_across_sessions():
    counter = {"active": 0, "max_active": 0, "lock": threading.Lock()}
    prototype = ChampsFinetunedInferenceAgent(
        menu=FINETUNED_MENU, adapter_path="x", model=FakeMedGemmaModel(counter))
    agent_a = prototype.create_session_agent()
    agent_b = prototype.create_session_agent()

    await asyncio.gather(
        agent_a.process_chunk("chunk-a", "one", []),
        agent_b.process_chunk("chunk-b", "two", []),
    )

    assert counter["max_active"] == 1


def test_render_profile_clinical_fields():
    profile = Metadata.from_dict({"profile": {
        "birth_weight_g": 2900, "hiv_status": "exposed",
        "weight_kg_admission": 6.9, "weight_kg_death": 6.5,
        "muac_cm_death": 11.2}}).profile

    assert render_profile(profile) == (
        "- birth weight: 2900 g\n- HIV status: exposed\n"
        "- weight: 6.9 kg at admission, 6.5 kg at death\n"
        "- MUAC: 11.2 cm at death\n")
