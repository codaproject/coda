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
from datetime import datetime
from typing import Dict, Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from coda import CODA_BASE
from coda.dialogue import (
    Transcriber,
    StreamingTranscriber,
    TRANSCRIBER_BACKENDS,
    create_transcriber,
)
from coda.dialogue.util import SPEECHMATICS_LANGUAGES
from coda.grounding.gilda_grounder import GildaGrounder
from coda.grounding.rag_grounder import RagGrounder
from coda.llm_api import create_llm_client
from coda.runtime_config import (
    get_grounder_type,
    get_inference_llm_model,
    get_inference_llm_provider,
    get_inference_url,
    get_rag_extractor_type,
    get_rag_llm_model,
    get_rag_llm_provider,
    get_rag_ontology,
    get_rag_use_reranker,
    get_transcriber_backend,
)

app = FastAPI()

# HTTP client for inference agent
INFERENCE_URL = get_inference_url()
inference_client = httpx.AsyncClient(base_url=INFERENCE_URL, timeout=120.0)

# Queue management for backpressure
MAX_PENDING_CHUNKS = 20
pending_chunks: Dict[str, asyncio.Task] = {}

# Streaming backends commit many tiny fragments; running an inference on each
# one floods the inference service. For streaming, accumulate committed text and
# infer once this many new words arrive, or after INFERENCE_MAX_WAIT_S if fewer
# words are still pending (so a short final utterance isn't left hanging). With
# no new text we don't infer at all. Chunked backends emit whole chunks and
# infer per chunk (threshold 0).
INFERENCE_MIN_WORDS = 15
INFERENCE_MAX_WAIT_S = 10.0

logger = logging.getLogger(__name__)

here = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(here, "templates")

# All languages supported by Whisper, keyed by ISO code
from whisper.tokenizer import LANGUAGES as _WHISPER_LANGUAGES

LANGUAGE_NAMES = {code: name.title() for code, name in _WHISPER_LANGUAGES.items()}

# Server-level settings
current_language = "en"
save_enabled = False
save_files: Dict[str, object] = {}  # open file handles keyed by language code
transcripts_dir = CODA_BASE.join(name="transcripts")
current_transcriber_backend = get_transcriber_backend()
current_whisper_model = "small"
current_llm_provider = get_inference_llm_provider()
current_llm_model = get_inference_llm_model()
current_grounder = get_grounder_type()
# RAG grounder settings, applied to the grounder via RagGrounder.update_config
rag_config = {
    "provider": get_rag_llm_provider(),
    "model": get_rag_llm_model(),
    "ontology": get_rag_ontology(),
    "use_reranker": get_rag_use_reranker(),
    "extractor_type": get_rag_extractor_type(),
}
# "whisper_translate" = use whisper task="translate" (direct speech-to-English)
# "llm" = transcribe in original language, then translate via LLM
translation_mode = "llm"
transcriber: Transcriber


class SettingsRequest(BaseModel):
    language: Optional[str] = None
    save_enabled: Optional[bool] = None
    transcriber_backend: Optional[str] = None
    whisper_model: Optional[str] = None
    grounder: Optional[str] = None
    rag_provider: Optional[str] = None
    rag_model: Optional[str] = None
    rag_ontology: Optional[str] = None
    rag_use_reranker: Optional[bool] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    translation_mode: Optional[str] = None


class UploadNoteRequest(BaseModel):
    text: str
    filename: Optional[str] = None
    clinical_note: Optional[str] = None


def get_language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code) or SPEECHMATICS_LANGUAGES.get(code, code)


def create_grounder(grounder_name: str):
    if grounder_name == "rag":
        grounder = RagGrounder()
        grounder.update_config(**rag_config)
        return grounder
    return GildaGrounder()


grounder = create_grounder(current_grounder)
transcriber = create_transcriber(
    current_transcriber_backend, whisper_model=current_whisper_model
)


def open_save_files(language: str):
    """Open transcript and annotation files for saving. Returns dict of file paths."""
    global save_files
    close_save_files()

    os.makedirs(transcripts_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    paths = {}

    if language != "en":
        # Original language file
        orig_path = os.path.join(transcripts_dir,
                                 f"transcript_{ts}_{language}.txt")
        save_files[language] = open(orig_path, "a", encoding="utf-8")
        paths[language] = orig_path

        # English translation file
        en_path = os.path.join(transcripts_dir, f"transcript_{ts}_en.txt")
        save_files["en"] = open(en_path, "a", encoding="utf-8")
        paths["en"] = en_path
    else:
        en_path = os.path.join(transcripts_dir, f"transcript_{ts}_en.txt")
        save_files["en"] = open(en_path, "a", encoding="utf-8")
        paths["en"] = en_path

    # Annotated dialogue file (JSON Lines - one JSON object per chunk)
    annotations_path = os.path.join(transcripts_dir,
                                    f"annotations_{ts}.jsonl")
    save_files["annotations"] = open(annotations_path, "a", encoding="utf-8")
    paths["annotations"] = annotations_path

    return paths


def close_save_files():
    """Close any open save files."""
    global save_files
    for f in save_files.values():
        try:
            f.close()
        except Exception:
            pass
    save_files.clear()


def save_transcript(text: str, lang_code: str):
    """Append a transcript line to the appropriate file."""
    f = save_files.get(lang_code)
    if f:
        f.write(text + "\n")
        f.flush()


def save_annotated_chunk(chunk_id: str, timestamp: float,
                         english_text: str, annotations,
                         original_text: str = None,
                         original_language: str = None):
    """Save a chunk with its annotations as a JSON Lines record."""
    f = save_files.get("annotations")
    if not f:
        return
    record = {
        "chunk_id": chunk_id,
        "timestamp": timestamp,
        "text": english_text,
        "annotations": [a.to_json() for a in annotations] if annotations else [],
    }
    if original_text:
        record["original_text"] = original_text
        record["original_language"] = original_language
    f.write(json.dumps(record) + "\n")
    f.flush()


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


async def process_inference(chunk_id: str, timestamp: float, transcript: str,
                            annotations: list, websocket: WebSocket):
    """Process inference in background and send results via HTTP."""
    try:
        # Send request to inference agent
        response = await inference_client.post("/infer", json={
            "chunk_id": chunk_id,
            "timestamp": timestamp,
            "text": transcript,
            "annotations": [a.to_json() for a in annotations]
        })
        response.raise_for_status()
        result = response.json()

        # Send inference result to client
        await _ws_send_safe(websocket, {"type": "inference", **result})
        # Log top cause
        causes = result.get('causes', {})
        if causes:
            top_curie = max(causes.items(), key=lambda x: x[1]['score'])[0]
            top_cause_name = causes[top_curie]['name']
            top_score = causes[top_curie]['score']
            logger.info(f"Inference result for {chunk_id}: {top_cause_name} ({top_curie}, score={top_score:.2f})")
        else:
            logger.info(f"Inference result for {chunk_id}: no causes")

    except httpx.TimeoutException:
        logger.error(f"Inference timeout for chunk {chunk_id}")
        await _ws_send_safe(websocket, {
            "type": "error", "chunk_id": chunk_id,
            "error": "Inference timeout"
        })
    except httpx.ConnectError:
        logger.error(f"Cannot connect to inference agent for chunk {chunk_id}")
        await _ws_send_safe(websocket, {
            "type": "error", "chunk_id": chunk_id,
            "error": "Inference agent unavailable"
        })
    except Exception as e:
        logger.error(f"Inference error for chunk {chunk_id}: {e}", exc_info=True)
        await _ws_send_safe(websocket, {
            "type": "error", "chunk_id": chunk_id,
            "error": str(e)
        })
    finally:
        # Clean up pending task
        if chunk_id in pending_chunks:
            del pending_chunks[chunk_id]


@app.get("/languages")
async def get_languages():
    """Get supported languages for the active transcription backend."""
    names = (SPEECHMATICS_LANGUAGES
             if current_transcriber_backend == "speechmatics"
             else LANGUAGE_NAMES)
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
    file_paths = {k: f.name for k, f in save_files.items()} if save_files else {}
    return {
        "language": current_language,
        "save_enabled": save_enabled,
        "file_paths": file_paths,
        "transcriber_backend": current_transcriber_backend,
        "whisper_model": current_whisper_model,
        "grounder": current_grounder,
        "rag_provider": rag_config["provider"],
        "rag_model": rag_config["model"],
        "rag_ontology": rag_config["ontology"],
        "rag_use_reranker": rag_config["use_reranker"],
        "llm_provider": current_llm_provider,
        "llm_model": current_llm_model,
        "translation_mode": translation_mode,
    }


@app.post("/settings")
async def update_settings(req: SettingsRequest):
    """Update server settings."""
    global current_language, save_enabled, transcriber, grounder
    global current_whisper_model, current_llm_provider, current_llm_model
    global translation_mode
    global current_grounder, current_transcriber_backend
    grounder_changed = False
    transcriber_changed = False
    if req.language is not None:
        current_language = req.language
        logger.info(f"Language set to: {current_language}")
    if req.save_enabled is not None:
        save_enabled = req.save_enabled
        if save_enabled:
            paths = open_save_files(current_language)
            logger.info(f"Transcript saving enabled: {paths}")
        else:
            close_save_files()
            logger.info("Transcript saving disabled")
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
            logger.info(f"Transcriber backend set to: {current_transcriber_backend}")
    if req.whisper_model is not None and req.whisper_model != current_whisper_model:
        current_whisper_model = req.whisper_model
        if current_transcriber_backend == "whisper":
            transcriber_changed = True
        logger.info(f"Whisper model set to: {current_whisper_model}")
    # Transcriber and grounder are independent; rebuild each only if it changed.
    if grounder_changed:
        grounder = await asyncio.to_thread(create_grounder, current_grounder)
        logger.info("Grounder reloaded: %s", current_grounder)
    if transcriber_changed:
        transcriber = await asyncio.to_thread(
            create_transcriber, current_transcriber_backend,
            current_whisper_model
        )
        logger.info(
            "Transcriber reloaded: backend=%s model=%s",
            current_transcriber_backend, current_whisper_model
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
    file_paths = {k: f.name for k, f in save_files.items()} if save_files else {}
    return {
        "language": current_language,
        "save_enabled": save_enabled,
        "file_paths": file_paths,
        "transcriber_backend": current_transcriber_backend,
        "whisper_model": current_whisper_model,
        "grounder": current_grounder,
        "rag_provider": rag_config["provider"],
        "rag_model": rag_config["model"],
        "rag_ontology": rag_config["ontology"],
        "rag_use_reranker": rag_config["use_reranker"],
        "llm_provider": current_llm_provider,
        "llm_model": current_llm_model,
        "translation_mode": translation_mode,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Capture and processing are decoupled into two tasks: one only drains the
    socket into an in-memory queue (so it is always drained and never
    overflows/disconnects), the other transcribes + grounds it at its own pace.
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    if save_enabled and not save_files:
        open_save_files(current_language)

    audio_queue: asyncio.Queue = asyncio.Queue()
    capture = asyncio.create_task(capture_audio(websocket, audio_queue))
    consume = asyncio.create_task(consume_transcripts(websocket, audio_queue))

    try:
        # `capture` surfaces WebSocketDisconnect when the user ends the session.
        await asyncio.gather(capture, consume)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        # Stop processing and any pending inference tasks.
        consume.cancel()
        for task in pending_chunks.values():
            task.cancel()
        pending_chunks.clear()
        capture.cancel()
        await asyncio.gather(capture, consume, return_exceptions=True)


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


async def consume_transcripts(websocket: WebSocket, queue: asyncio.Queue):
    """Process: consume transcript events from the active transcriber and, for
    each committed event, translate, ground, save, and display it. Inference
    runs on the accumulated text once enough has arrived (see INFERENCE_MIN_WORDS).
    """
    async def audio_iter():
        while True:
            data = await queue.get()
            if data is None:
                return
            yield data

    # Direct speech-to-English translation is a Whisper capability; for other
    # backends, non-English transcribes then translates via the LLM. Captured at
    # connection start (settings changes apply on the next connection).
    direct_translate = (current_language != "en"
                        and translation_mode == "whisper_translate"
                        and current_transcriber_backend == "whisper")
    task = "translate" if direct_translate else "transcribe"

    # Committed text accumulates here until enough has arrived to infer on; see
    # INFERENCE_MIN_WORDS. Chunked backends use threshold 0 (infer per chunk).
    threshold = (INFERENCE_MIN_WORDS
                 if isinstance(transcriber, StreamingTranscriber) else 0)
    pending = {"text": [], "anns": [], "words": 0,
               "chunk_id": None, "timestamp": None}
    last_infer = time.monotonic()

    async def flush():
        nonlocal last_infer
        if not pending["text"]:
            return
        text = " ".join(pending["text"])
        anns, chunk_id, timestamp = (pending["anns"], pending["chunk_id"],
                                     pending["timestamp"])
        pending.update(text=[], anns=[], words=0)
        last_infer = time.monotonic()
        await _start_inference(websocket, chunk_id, timestamp, text, anns)

    async def idle_flush():
        # Flush pending text that never reached the word threshold once it has
        # waited long enough, so a short trailing utterance still gets inferred.
        while True:
            await asyncio.sleep(1.0)
            if pending["text"] and \
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
                committed = await _handle_committed(websocket, event,
                                                    direct_translate)
            except Exception as e:
                logger.error(f"Error on event {event.id}: {e}", exc_info=True)
                continue
            if committed is None:
                continue
            chunk_id, timestamp, text, anns = committed
            pending["text"].append(text)
            pending["anns"].extend(anns)
            pending["words"] += len(text.split())
            pending["chunk_id"] = chunk_id
            pending["timestamp"] = timestamp
            if pending["words"] >= threshold:
                await flush()
        await flush()
    finally:
        timer.cancel()


async def _handle_committed(websocket: WebSocket, event, direct_translate: bool):
    """Translate, ground, save, and display one committed transcript event.

    Returns (chunk_id, timestamp, english_text, annotations) for the caller to
    accumulate toward inference, or None if there was no usable text.
    """
    chunk_id = event.id
    timestamp = event.timestamp
    original_transcript = None
    english_text = event.text

    # If non-English and not already translated to English, translate via LLM
    # (skip if transcript is too short to be real speech).
    if (not direct_translate and current_language != "en"
            and len(event.text.split()) > 1):
        original_transcript = event.text
        english_text = await translate_text(event.text, current_language)

    # Ground the (final, English) text without blocking the loop
    annotations = []
    if english_text:
        annotations = await asyncio.to_thread(grounder.annotate, english_text)

    if not english_text:
        return None

    # Save transcripts and annotations if enabled
    if save_enabled:
        save_transcript(english_text, "en")
        if original_transcript and current_language != "en":
            save_transcript(original_transcript, current_language)
        save_annotated_chunk(
            chunk_id, timestamp, english_text, annotations,
            original_text=original_transcript,
            original_language=(current_language
                               if current_language != "en" else None),
        )

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
    await websocket.send_json(msg)
    logger.info(f"Chunk {chunk_id}: {english_text}")

    return chunk_id, timestamp, english_text, annotations


async def _start_inference(websocket: WebSocket, chunk_id: str, timestamp: float,
                           text: str, annotations: list):
    """Launch a background inference on accumulated committed text.

    Gating upstream keeps the rate sane; backpressure here is a last resort that
    drops the oldest pending inference if the service still falls behind."""
    if len(pending_chunks) >= MAX_PENDING_CHUNKS:
        oldest_id = next(iter(pending_chunks))
        pending_chunks[oldest_id].cancel()
        del pending_chunks[oldest_id]
        logger.warning(f"Dropped inference {oldest_id} due to backpressure")
        await _ws_send_safe(websocket, {
            "type": "warning",
            "message": "Processing slower than audio - dropping old chunks"
        })

    inference_task = asyncio.create_task(
        process_inference(chunk_id, timestamp, text, annotations, websocket)
    )
    pending_chunks[chunk_id] = inference_task


MAX_NOTE_SIZE = 100_000  # 100KB limit for clinical notes


@app.post("/upload-note")
async def upload_note(req: UploadNoteRequest):
    """Upload a clinical note for grounding and inference."""
    if not req.text.strip():
        return JSONResponse(status_code=400,
                            content={"error": "Empty clinical note"})
    if len(req.text) > MAX_NOTE_SIZE:
        return JSONResponse(status_code=413,
                            content={"error": "Clinical note too large (max 100KB)"})

    chunk_id = "note-" + str(uuid.uuid4())
    timestamp = time.time()

    # Run grounding in the dedicated executor for SQLite thread safety
    # Adapted since the grounder and transcriber have been separated
    annotations = await asyncio.to_thread(
        grounder.annotate,
        req.text.strip(),
    )

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

    # Save if enabled
    if save_enabled:
        if not save_files:
            open_save_files(current_language)
        save_transcript(req.text.strip(), "en")
        save_annotated_chunk(chunk_id, timestamp, req.text.strip(), annotations)

    # Send to inference agent
    inference_result = None
    try:
        response = await inference_client.post("/infer", json={
            "chunk_id": chunk_id,
            "timestamp": timestamp,
            "text": "",  # Note is not a transcript chunk
            "annotations": [a.to_json() for a in annotations] if annotations else [],
            "clinical_note": req.text.strip(),
        })
        response.raise_for_status()
        inference_result = response.json()
    except Exception as e:
        logger.error(f"Inference error for uploaded note: {e}", exc_info=True)
        inference_result = {"error": str(e)}

    return {
        "chunk_id": chunk_id,
        "timestamp": timestamp,
        "annotations": structured_annotations,
        "inference": inference_result,
        "filename": req.filename,
    }


@app.post("/reset")
async def reset_session():
    """Reset session state: close save files and reset the inference agent."""
    close_save_files()
    try:
        resp = await inference_client.post("/reset")
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
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
