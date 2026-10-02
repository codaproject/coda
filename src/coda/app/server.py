"""
Real-time Voice Transcription Server using OpenAI Whisper
Requirements:
    pip install fastapi uvicorn websockets whisper numpy scipy

To run:
    python server.py
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import httpx
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from coda.app.storage import CaseRecorder, storage_enabled
from coda.app.onboarding_notice import (
    load_onboarding_notice_html,
    render_onboarding_notice,
)
from coda.dialogue import (
    Transcriber,
    StreamingTranscriber,
    TRANSCRIBER_BACKENDS,
    create_transcriber,
    get_transcriber_models,
    get_transcriber_languages,
)
from coda.inference.streaming import (
    INFERENCE_MAX_WAIT_S,
    INFERENCE_MIN_WORDS,
    StreamingInferenceBuffer,
)
from coda.grounding.gilda_grounder import GildaGrounder
from coda.grounding.rag_grounder import RagGrounder
from coda.llm_api import create_llm_client
from coda.config import settings, inference_url
from coda.metadata import Metadata

app = FastAPI()

# HTTP client for inference agent
INFERENCE_URL = inference_url()
inference_client = httpx.AsyncClient(base_url=INFERENCE_URL, timeout=120.0)

logger = logging.getLogger(__name__)

here = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(here, "templates")

# Server-level settings
current_language = settings.dialogue.language
current_transcriber_backend = settings.dialogue.transcriber_backend


def _default_model_for(backend: str):
    """The backend's default model, or None if the backend can't be loaded."""
    try:
        return get_transcriber_models(backend)["default_model"]
    except Exception as e:
        logger.warning("Transcriber backend %r unavailable: %s", backend, e)
        return None


current_transcriber_model = settings.dialogue.transcriber_model \
    or _default_model_for(current_transcriber_backend)
current_llm_provider = settings.inference.llm.provider
current_llm_model = settings.inference.llm.model
current_grounder = settings.grounder.type
# RAG grounder settings, applied to the grounder via RagGrounder.update_config
rag_config = {
    "provider": settings.grounder.rag.llm.provider,
    "model": settings.grounder.rag.llm.model,
    "ontology": settings.grounder.rag.retriever.ontology,
    "use_reranker": settings.grounder.rag.reranker.enabled,
    "extractor_type": settings.grounder.rag.extractor.type,
}
# "whisper_translate" = use whisper task="translate" (direct speech-to-English)
# "llm" = transcribe in original language, then translate via LLM
translation_mode = settings.dialogue.translation_mode
# Per-interview metadata, set via /metadata and forwarded to the inference
# agent with every inference request.
current_metadata = Metadata()
transcriber: Transcriber
active_inference_sessions: set["InferenceSessionCoordinator"] = set()


class SettingsRequest(BaseModel):
    language: Optional[str] = None
    transcriber_backend: Optional[str] = None
    transcriber_model: Optional[str] = None
    grounder: Optional[str] = None
    rag_provider: Optional[str] = None
    rag_model: Optional[str] = None
    rag_ontology: Optional[str] = None
    rag_use_reranker: Optional[bool] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    translation_mode: Optional[str] = None


@dataclass
class PendingInferenceChunk:
    chunk_id: str
    timestamp: float
    text: str
    annotations: list
    queued_at: float = field(default_factory=time.perf_counter)


@dataclass
class CoalescedInferenceBatch:
    chunk_id: str
    timestamp: float
    text: str
    annotations: list
    queued_at: float
    chunk_count: int


class InferenceSessionCoordinator:
    """Per-WebSocket inference scheduler with request coalescing."""

    def __init__(self, websocket: WebSocket, session_id: Optional[str] = None):
        self.websocket = websocket
        self.session_id = session_id or str(uuid.uuid4())
        self.generation = 0
        self._lock = asyncio.Lock()
        self._pending: list[PendingInferenceChunk] = []
        self._drain_task: asyncio.Task | None = None
        self.recorder: CaseRecorder | None = None

    async def start_case(self):
        """Begin recording a new case for the current generation, if enabled."""
        if not storage_enabled():
            return
        recorder = CaseRecorder(self.session_id, self.generation,
                                await _run_info())
        self.end_case()
        self.recorder = recorder
        logger.info("Recording case %s to %s", recorder.case_id,
                    recorder.case_dir)

    def end_case(self):
        if self.recorder is not None:
            self.recorder.close(current_metadata.to_dict())
            logger.info("Closed case %s", self.recorder.case_id)
            self.recorder = None

    async def enqueue(self, chunk_id: str, timestamp: float, text: str,
                      annotations: list):
        if not text:
            return
        async with self._lock:
            self._pending.append(PendingInferenceChunk(
                chunk_id=chunk_id,
                timestamp=timestamp,
                text=text,
                annotations=list(annotations),
            ))
            if self._drain_task is None or self._drain_task.done():
                self._drain_task = asyncio.create_task(self._drain_loop())

    async def invalidate(self, reason: str):
        async with self._lock:
            dropped = len(self._pending)
            old_generation = self.generation
            self.generation += 1
            self._pending.clear()
        logger.info(
            "Invalidated inference session %s generation %d (%s, cleared %d buffered chunk(s))",
            self.session_id, old_generation, reason, dropped
        )
        await _reset_inference_session(self.session_id, old_generation)
        if reason == "disconnect":
            self.end_case()
        else:
            await self.start_case()

    async def wait_for_idle(self):
        task = None
        async with self._lock:
            task = self._drain_task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    async def _drain_loop(self):
        while True:
            async with self._lock:
                request_generation = self.generation
                batch = self._take_pending_batch_locked()
                if batch is None:
                    self._drain_task = None
                    return
            buffer_wait_s = time.perf_counter() - batch.queued_at
            logger.info(
                "Dispatching inference batch for session %s generation %d: chunks=%d text_chars=%d buffer_wait=%.2fs latest_chunk=%s",
                self.session_id, request_generation, batch.chunk_count,
                len(batch.text), buffer_wait_s, batch.chunk_id
            )
            await process_inference(
                self,
                request_generation=request_generation,
                batch=batch,
                buffer_wait_s=buffer_wait_s,
            )

    def _take_pending_batch_locked(self) -> CoalescedInferenceBatch | None:
        if not self._pending:
            return None
        chunks = self._pending
        self._pending = []
        newest = chunks[-1]
        text = " ".join(chunk.text for chunk in chunks)
        annotations = [
            ann
            for chunk in chunks
            for ann in chunk.annotations
        ]
        return CoalescedInferenceBatch(
            chunk_id=newest.chunk_id,
            timestamp=newest.timestamp,
            text=text,
            annotations=annotations,
            queued_at=chunks[0].queued_at,
            chunk_count=len(chunks),
        )

    async def is_current_generation(self, generation: int) -> bool:
        async with self._lock:
            return self.generation == generation


class ResetRequest(BaseModel):
    session_id: Optional[str] = None
    session_generation: Optional[int] = None


def get_language_name(code: str) -> str:
    return get_transcriber_languages(current_transcriber_backend).get(code, code)


def create_grounder(grounder_name: str):
    if grounder_name == "rag":
        grounder = RagGrounder()
        grounder.update_config(**rag_config)
        return grounder
    return GildaGrounder()


grounder = create_grounder(current_grounder)
transcriber = create_transcriber(
    current_transcriber_backend, model=current_transcriber_model
)
if transcriber.normalize_language(current_language) is None:
    # A backend that can't do the configured language returns empty text for
    # every chunk.
    logger.warning(
        "Transcriber backend %r does not support language %r",
        current_transcriber_backend, current_language,
    )


async def translate_text(text: str, source_language: str) -> str:
    """Translate text to English using the LLM API."""
    lang_name = get_language_name(source_language)
    prompt = (f"Translate the following {lang_name} text to English. "
              f"Return only the translation, nothing else.\n\n{text}")
    try:
        llm = create_llm_client(provider=current_llm_provider,
                                model=current_llm_model)
        translation = await asyncio.to_thread(llm.call, prompt)
        return translation.strip()
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text  # fall back to original text


def render_annotations(annotations):
    """Render annotations as a list of strings."""
    if not annotations:
        return []
    parts = []
    for ann in annotations:
        term = ann.matches[0].term
        curie = term.get_curie()
        name = term.entry_name
        text = ann.text
        part = f"{text} = {curie} ({name})"
        parts.append(part)
    return parts


async def _ws_send_safe(websocket: WebSocket, data: dict):
    """Send JSON over WebSocket, silently ignoring disconnected clients."""
    try:
        await websocket.send_json(data)
    except (WebSocketDisconnect, RuntimeError):
        pass


async def _reset_inference_session(session_id: str, generation: int):
    try:
        resp = await inference_client.post("/reset", json={
            "session_id": session_id,
            "session_generation": generation,
        })
        resp.raise_for_status()
        logger.info(
            "Inference agent reset for session %s generation %d",
            session_id, generation
        )
    except Exception as e:
        logger.warning(
            "Could not reset inference session %s generation %d: %s",
            session_id, generation, e
        )


async def _run_info() -> dict:
    """Settings and model versions in effect, recorded with each case."""
    try:
        resp = await inference_client.get("/info", timeout=5.0)
        resp.raise_for_status()
        inference_info = resp.json()
    except Exception as e:
        logger.warning("Could not fetch inference service info: %s", e)
        inference_info = None
    return {
        "language": current_language,
        "translation_mode": translation_mode,
        "transcriber_backend": current_transcriber_backend,
        "transcriber_model": current_transcriber_model,
        "grounder": current_grounder,
        "rag": rag_config if current_grounder == "rag" else None,
        "translation_llm": {"provider": current_llm_provider,
                            "model": current_llm_model},
        "inference": inference_info,
    }


async def process_inference(session: InferenceSessionCoordinator,
                            request_generation: int,
                            batch: CoalescedInferenceBatch,
                            buffer_wait_s: float):
    """Process inference in background and send results via HTTP."""
    started = time.perf_counter()
    try:
        # Send request to inference agent
        request = {
            "chunk_id": batch.chunk_id,
            "timestamp": batch.timestamp,
            "text": batch.text,
            "annotations": [a.to_json() for a in batch.annotations],
            "metadata": current_metadata.to_dict(),
            "session_id": session.session_id,
            "session_generation": request_generation,
        }
        response = await inference_client.post("/infer", json=request)
        response.raise_for_status()
        result = response.json()
        infer_s = time.perf_counter() - started
        result["timings"] = {
            **result.get("timings", {}),
            "request_s": round(infer_s, 3),
            "buffer_wait_s": round(buffer_wait_s, 3),
        }

        if not await session.is_current_generation(request_generation):
            logger.info(
                "Discarded stale inference result for session %s generation %d after %.2fs",
                session.session_id, request_generation, infer_s
            )
            return

        if session.recorder is not None:
            session.recorder.write_inference(request, result,
                                             shown_at=time.time())
        # Send inference result to client
        await _ws_send_safe(session.websocket, {"type": "inference", **result})
        # Log top cause
        causes = result.get('causes', {})
        if causes:
            top_curie = max(causes.items(), key=lambda x: x[1]['score'])[0]
            top_cause_name = causes[top_curie]['name']
            top_score = causes[top_curie]['score']
            logger.info(
                "Inference result for %s in %.2fs: %s (%s, score=%.2f, chunks=%d, text_chars=%d, wait=%.2fs)",
                batch.chunk_id, infer_s, top_cause_name, top_curie, top_score,
                batch.chunk_count, len(batch.text), buffer_wait_s
            )
        else:
            logger.info(
                "Inference result for %s in %.2fs: no causes (chunks=%d, text_chars=%d, wait=%.2fs)",
                batch.chunk_id, infer_s, batch.chunk_count, len(batch.text),
                buffer_wait_s
            )

    except httpx.TimeoutException:
        logger.error("Inference timeout for chunk %s after %.2fs",
                     batch.chunk_id, time.perf_counter() - started)
        if await session.is_current_generation(request_generation):
            await _ws_send_safe(session.websocket, {
                "type": "error", "chunk_id": batch.chunk_id,
                "error": "Inference timeout"
            })
    except httpx.ConnectError:
        logger.error("Cannot connect to inference agent for chunk %s after %.2fs",
                     batch.chunk_id, time.perf_counter() - started)
        if await session.is_current_generation(request_generation):
            await _ws_send_safe(session.websocket, {
                "type": "error", "chunk_id": batch.chunk_id,
                "error": "Inference agent unavailable"
            })
    except Exception as e:
        logger.error("Inference error for chunk %s after %.2fs: %s",
                     batch.chunk_id, time.perf_counter() - started, e, exc_info=True)
        if await session.is_current_generation(request_generation):
            await _ws_send_safe(session.websocket, {
                "type": "error", "chunk_id": batch.chunk_id,
                "error": str(e)
            })


@app.get("/languages")
async def get_languages():
    """Get supported languages for the active transcription backend."""
    names = get_transcriber_languages(current_transcriber_backend)
    # Return sorted by name, with English first
    langs = [{"code": code, "name": name}
             for code, name in sorted(names.items(), key=lambda x: x[1])]
    # Move English to front
    langs = ([l for l in langs if l["code"] == "en"]
             + [l for l in langs if l["code"] != "en"])
    return langs


@app.get("/settings")
async def get_settings():
    """Get current server settings."""
    return {
        "language": current_language,
        "storage_enabled": storage_enabled(),
        "transcriber_backend": current_transcriber_backend,
        "transcriber_model": current_transcriber_model,
        "grounder": current_grounder,
        "rag_provider": rag_config["provider"],
        "rag_model": rag_config["model"],
        "rag_ontology": rag_config["ontology"],
        "rag_use_reranker": rag_config["use_reranker"],
        "llm_provider": current_llm_provider,
        "llm_model": current_llm_model,
        "translation_mode": translation_mode,
        "server_settings_locked": settings.app.get("lock_server_settings", False),
    }


@app.post("/metadata")
async def set_metadata(payload: dict = Body(...)):
    """Set per-interview metadata forwarded to the inference agent."""
    global current_metadata
    current_metadata = Metadata.from_dict(payload)
    logger.info(f"Metadata set: {current_metadata.to_dict()}")
    return current_metadata.to_dict()


@app.get("/transcriber_backends")
async def get_transcriber_backends():
    """List selectable transcription backends for the settings UI."""
    return {"backends": list(TRANSCRIBER_BACKENDS)}


@app.get("/transcriber_backends/{backend}")
async def get_transcriber_backend_models(backend: str):
    """Return one backend's selectable models, loaded on demand.

    The backend is imported only here; if its dependencies aren't installed the
    response reports it as unavailable so the UI can warn instead of failing.
    """
    if backend not in TRANSCRIBER_BACKENDS:
        raise HTTPException(status_code=404,
                            detail=f"Unknown backend {backend!r}")
    try:
        info = await asyncio.to_thread(get_transcriber_models, backend)
    except Exception as e:
        logger.warning("Transcriber backend %r unavailable: %s", backend, e)
        return {"backend": backend, "available": False, "error": str(e)}
    return {"backend": backend, "available": True, **info}


@app.post("/settings")
async def update_settings(req: SettingsRequest):
    """Update server settings."""
    if settings.app.get("lock_server_settings", False):
        raise HTTPException(status_code=403, detail="Server settings are locked")
    global current_language, transcriber, grounder
    global current_transcriber_model, current_llm_provider, current_llm_model
    global translation_mode
    global current_grounder, current_transcriber_backend
    grounder_changed = False
    transcriber_changed = False
    prev_backend = current_transcriber_backend
    prev_model = current_transcriber_model
    if req.language is not None:
        current_language = req.language
        logger.info(f"Language set to: {current_language}")
    if req.grounder is not None:
        grounder_name = req.grounder.strip().lower()
        if grounder_name not in {"gilda", "rag"}:
            grounder_name = "gilda"
        if grounder_name != current_grounder:
            current_grounder = grounder_name
            grounder_changed = True
            logger.info(f"Grounder set to: {current_grounder}")
    rag_updated = False
    if req.rag_provider is not None:
        rag_config["provider"] = req.rag_provider
        rag_updated = True
    if req.rag_model is not None:
        rag_config["model"] = req.rag_model
        rag_updated = True
    if req.rag_ontology is not None:
        rag_config["ontology"] = req.rag_ontology
        rag_updated = True
    if req.rag_use_reranker is not None:
        rag_config["use_reranker"] = req.rag_use_reranker
        rag_updated = True
    if rag_updated:
        if isinstance(grounder, RagGrounder):
            await asyncio.to_thread(grounder.update_config, **rag_config)
        logger.info(f"RAG grounder config updated: {rag_config}")
    if req.transcriber_backend is not None:
        backend = req.transcriber_backend.strip().lower()
        if backend not in TRANSCRIBER_BACKENDS:
            backend = current_transcriber_backend
        if backend != current_transcriber_backend:
            current_transcriber_backend = backend
            transcriber_changed = True
            if req.transcriber_model is None:
                current_transcriber_model = await asyncio.to_thread(
                    _default_model_for, backend)
            logger.info(f"Transcriber backend set to: {current_transcriber_backend}")
    if (req.transcriber_model is not None
            and req.transcriber_model != current_transcriber_model):
        current_transcriber_model = req.transcriber_model
        transcriber_changed = True
        logger.info(f"Transcriber model set to: {current_transcriber_model}")
    # Transcriber and grounder are independent; rebuild each only if it changed.
    if grounder_changed:
        grounder = await asyncio.to_thread(create_grounder, current_grounder)
        logger.info("Grounder reloaded: %s", current_grounder)
    if transcriber_changed:
        try:
            transcriber = await asyncio.to_thread(
                create_transcriber, current_transcriber_backend,
                current_transcriber_model
            )
        except Exception as e:
            current_transcriber_backend = prev_backend
            current_transcriber_model = prev_model
            logger.error("Failed to load transcriber: %s", e)
            raise HTTPException(
                status_code=400,
                detail=f"Could not load transcriber: {e}") from e
        logger.info(
            "Transcriber reloaded: backend=%s model=%s",
            current_transcriber_backend, current_transcriber_model
        )
    if req.llm_provider is not None:
        current_llm_provider = req.llm_provider
        logger.info(f"LLM provider set to: {current_llm_provider}")
    if req.llm_model is not None:
        current_llm_model = req.llm_model
        logger.info(f"LLM model set to: {current_llm_model}")
    if req.translation_mode is not None:
        translation_mode = req.translation_mode
        logger.info(f"Translation mode set to: {translation_mode}")
    return {
        "language": current_language,
        "storage_enabled": storage_enabled(),
        "transcriber_backend": current_transcriber_backend,
        "transcriber_model": current_transcriber_model,
        "grounder": current_grounder,
        "rag_provider": rag_config["provider"],
        "rag_model": rag_config["model"],
        "rag_ontology": rag_config["ontology"],
        "rag_use_reranker": rag_config["use_reranker"],
        "llm_provider": current_llm_provider,
        "llm_model": current_llm_model,
        "translation_mode": translation_mode,
        "server_settings_locked": settings.app.get("lock_server_settings", False),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Capture and processing are decoupled into two tasks: one only drains the
    socket into an in-memory queue (so it is always drained and never
    overflows/disconnects), the other transcribes + grounds it at its own pace.
    """
    await websocket.accept()
    logger.info("WebSocket connection established")

    inference_session = InferenceSessionCoordinator(websocket)
    active_inference_sessions.add(inference_session)
    await inference_session.start_case()
    await _ws_send_safe(websocket, {
        "type": "session",
        "session_id": inference_session.session_id,
        "session_generation": inference_session.generation,
    })
    audio_queue: asyncio.Queue = asyncio.Queue()
    capture = asyncio.create_task(capture_audio(websocket, audio_queue))
    consume = asyncio.create_task(
        consume_transcripts(websocket, audio_queue, inference_session)
    )

    try:
        # `capture` surfaces WebSocketDisconnect when the user ends the session.
        await asyncio.gather(capture, consume)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        consume.cancel()
        capture.cancel()
        await asyncio.gather(capture, consume, return_exceptions=True)
        await inference_session.invalidate("disconnect")
        await inference_session.wait_for_idle()
        active_inference_sessions.discard(inference_session)


async def capture_audio(websocket: WebSocket, queue: asyncio.Queue):
    """Capture: drain the socket into the queue. Never blocks transcription.

    Surfaces WebSocketDisconnect to the gather; the sentinel lets the consumer's
    audio iterator (and thus the transcriber stream) end cleanly.
    """
    try:
        while True:
            queue.put_nowait(await websocket.receive_bytes())
    finally:
        queue.put_nowait(None)


async def consume_transcripts(websocket: WebSocket, queue: asyncio.Queue,
                              inference_session: InferenceSessionCoordinator):
    """Process: consume transcript events from the active transcriber and, for
    each committed event, translate, ground, save, and display it. Inference
    runs on the accumulated text once enough has arrived (see INFERENCE_MIN_WORDS).
    """

    async def audio_iter():
        while True:
            data = await queue.get()
            if data is None:
                return
            if inference_session.recorder is not None:
                inference_session.recorder.write_audio(data)
            yield data

    # Direct speech-to-English translation is a Whisper capability; for other
    # backends, non-English transcribes then translates via the LLM. Captured at
    # connection start (settings changes apply on the next connection).
    direct_translate = (current_language != "en"
                        and translation_mode == "whisper_translate"
                        and current_transcriber_backend == "whisper")
    task = "translate" if direct_translate else "transcribe"

    # Committed text accumulates until enough has arrived to infer on. Streaming
    # backends use INFERENCE_MIN_WORDS; chunked backends emit whole chunks and
    # infer per chunk (min_words=0).
    buf = StreamingInferenceBuffer(
        min_words=INFERENCE_MIN_WORDS
        if isinstance(transcriber, StreamingTranscriber) else 0)
    last_infer = time.monotonic()

    async def flush():
        nonlocal last_infer
        batch = buf.take()
        if batch is None:
            return
        text, anns, chunk_id, timestamp = batch
        last_infer = time.monotonic()
        await _start_inference(inference_session, chunk_id, timestamp, text, anns)

    async def idle_flush():
        # Flush pending text that never reached the word threshold once it has
        # waited long enough, so a short trailing utterance still gets inferred.
        while True:
            await asyncio.sleep(1.0)
            if buf.has_pending and \
                    time.monotonic() - last_infer >= INFERENCE_MAX_WAIT_S:
                await flush()

    timer = asyncio.create_task(idle_flush())
    try:
        async for event in transcriber.stream(
                audio_iter(), language=current_language, task=task):
            if not event.committed:
                await _ws_send_safe(websocket,
                                    {"type": "preview", "text": event.text})
                continue
            # One bad event shouldn't kill the session.
            try:
                committed = await _handle_committed(
                    inference_session, event, direct_translate)
            except Exception as e:
                logger.error(f"Error on event {event.id}: {e}", exc_info=True)
                continue
            if committed is None:
                continue
            chunk_id, timestamp, text, anns = committed
            buf.add(text, anns, chunk_id, timestamp)
            if buf.ready:
                await flush()
        await flush()
    finally:
        timer.cancel()


async def _handle_committed(inference_session: InferenceSessionCoordinator,
                            event, direct_translate: bool):
    """Translate, ground, save, and display one committed transcript event.

    Returns (chunk_id, timestamp, english_text, annotations) for the caller to
    accumulate toward inference, or None if there was no usable text.
    """
    chunk_id = event.id
    timestamp = event.timestamp
    original_transcript = None
    english_text = event.text
    total_start = time.perf_counter()
    translation_s = 0.0
    grounding_s = 0.0
    save_s = 0.0
    emit_s = 0.0

    # If non-English and not already translated to English, translate via LLM
    # (skip if transcript is too short to be real speech).
    if (not direct_translate and current_language != "en"
            and len(event.text.split()) > 1):
        original_transcript = event.text
        translation_start = time.perf_counter()
        english_text = await translate_text(event.text, current_language)
        translation_s = time.perf_counter() - translation_start

    # Ground the (final, English) text without blocking the loop
    annotations = []
    if english_text:
        grounding_start = time.perf_counter()
        annotations = await asyncio.to_thread(grounder.annotate, english_text)
        grounding_s = time.perf_counter() - grounding_start

    if not english_text:
        return None

    recorder = inference_session.recorder
    if recorder is not None:
        save_start = time.perf_counter()
        recorder.write_chunk(
            chunk_id, timestamp, english_text, annotations,
            timings={"translate_s": round(translation_s, 3),
                     "ground_s": round(grounding_s, 3)},
            original_text=original_transcript,
            original_language=(current_language
                               if original_transcript else None),
        )
        save_s = time.perf_counter() - save_start

    # Build structured annotations for inline display
    structured_annotations = [
        {
            "text": ann.text,
            "start": ann.start,
            "end": ann.end,
            "curie": ann.matches[0].term.get_curie(),
            "name": ann.matches[0].term.entry_name,
        }
        for ann in annotations
    ] if annotations else []

    # Send transcript to client
    msg = {
        "type": "transcript",
        "chunk_id": chunk_id,
        "timestamp": timestamp,
        "transcript": english_text,
        "annotations": structured_annotations,
    }
    if original_transcript:
        msg["original_transcript"] = original_transcript
        msg["original_language"] = current_language
    emit_start = time.perf_counter()
    await _ws_send_safe(inference_session.websocket, msg)
    emit_s = time.perf_counter() - emit_start
    total_s = time.perf_counter() - total_start
    logger.info(
        "Chunk %s processed in %.2fs (translate=%.2fs ground=%.2fs save=%.2fs emit=%.2fs text_chars=%d annotations=%d)",
        chunk_id, total_s, translation_s, grounding_s, save_s, emit_s,
        len(english_text), len(annotations)
    )
    logger.info(f"Chunk {chunk_id}: {english_text}")

    return chunk_id, timestamp, english_text, annotations


async def _start_inference(inference_session: InferenceSessionCoordinator,
                           chunk_id: str, timestamp: float,
                           text: str, annotations: list):
    """Queue a chunk for coalesced inference dispatch."""
    await inference_session.enqueue(chunk_id, timestamp, text, annotations)


@app.post("/reset")
async def reset_session(req: Optional[ResetRequest] = None):
    """Reset session state: start a new case and reset the inference agent."""
    global current_metadata
    targeted_reset = req is not None and req.session_id is not None
    if not targeted_reset:
        current_metadata = Metadata()
    target_sessions = list(active_inference_sessions)
    if targeted_reset:
        target_sessions = [
            session for session in target_sessions
            if session.session_id == req.session_id
        ]
    await asyncio.gather(
        *(session.invalidate("reset") for session in target_sessions),
        return_exceptions=True,
    )
    if not target_sessions:
        try:
            resp = await inference_client.post("/reset", json=req.model_dump() if req else {})
            resp.raise_for_status()
            logger.info("Inference agent reset")
        except Exception as e:
            logger.warning(f"Could not reset inference agent: {e}")
    return {"status": "reset"}


@app.get("/")
async def get_index():
    """Serve the index page."""
    with open(os.path.join(templates_dir, "index.html"), "r") as fh:
        html_content = fh.read()
    notice_html = load_onboarding_notice_html(
        settings.app.onboarding_notice.file,
    )
    html_content = render_onboarding_notice(
        html_content,
        notice_html=notice_html,
        notice_version=settings.app.onboarding_notice.version,
    )
    html_content = html_content.replace(
        "__CODA_UI_LANGUAGE__", json.dumps(settings.app.ui_language),
    )
    if not settings.app.get("show_case_profile", True):
        html_content = html_content.replace(
            '<div class="profile-bar" id="profileBar" data-expanded="false">',
            '<div class="profile-bar" id="profileBar" data-expanded="false" hidden>',
        )
    if not settings.app.get("show_cod_result", True):
        html_content = html_content.replace(
            "<body",
            '<body data-show-cod-result="false"',
            1,
        )
    return HTMLResponse(content=html_content)


@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
