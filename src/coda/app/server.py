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
import shutil
import uuid
from datetime import datetime
from typing import Dict, Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from coda import CODA_BASE
from coda.dialogue.whisper import WhisperTranscriber
from coda.dialogue import AudioProcessor
from coda.grounding.gilda_grounder import GildaGrounder
from pathlib import Path

from coda.grounding.rag_grounder import RagGrounder
from coda.llm_api import create_llm_client

app = FastAPI()


# HTTP client for inference agent
INFERENCE_URL = os.getenv("CODA_INFERENCE_URL", "http://localhost:5123")
inference_client = httpx.AsyncClient(base_url=INFERENCE_URL, timeout=120.0)

# Queue management for backpressure
MAX_PENDING_CHUNKS = 10
pending_chunks: Dict[str, asyncio.Task] = {}

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
current_whisper_model = "medium"
current_grounder = "gilda"
# RAG grounder settings, applied to the grounder via RagGrounder.update_config
rag_config = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "ontology": "icd10",
    "use_reranker": True,
}
# Translation LLM settings (separate from the RAG grounder's LLM)
current_llm_provider = "openai"
current_llm_model = "gpt-5.4-mini"
# "whisper_translate" = use whisper task="translate" (direct speech-to-English)
# "llm" = transcribe in original language, then translate via LLM
translation_mode = "llm"
transcriber: WhisperTranscriber


class SettingsRequest(BaseModel):
    language: Optional[str] = None
    save_enabled: Optional[bool] = None
    whisper_model: Optional[str] = None
    grounder: Optional[str] = None
    rag_provider: Optional[str] = None
    rag_model: Optional[str] = None
    rag_ontology: Optional[str] = None
    rag_use_reranker: Optional[bool] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    translation_mode: Optional[str] = None


def get_language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def create_grounder(grounder_name: str):
    if grounder_name == "rag":
        grounder = RagGrounder()
        grounder.update_config(**rag_config)
        return grounder
    return GildaGrounder()


grounder = create_grounder(current_grounder)
transcriber = WhisperTranscriber(model_size=current_whisper_model)


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
    """Get all supported languages."""
    # Return sorted by name, with English first
    langs = [{"code": code, "name": name}
             for code, name in sorted(LANGUAGE_NAMES.items(),
                                      key=lambda x: x[1])]
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
    global current_grounder
    grounder_changed = False
    whisper_changed = False
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
    if req.whisper_model is not None and req.whisper_model != current_whisper_model:
        current_whisper_model = req.whisper_model
        whisper_changed = True
        logger.info(f"Whisper model set to: {current_whisper_model}")
    # Transcriber and grounder are independent; rebuild each only if it changed.
    if grounder_changed:
        grounder = await asyncio.to_thread(create_grounder, current_grounder)
        logger.info("Grounder reloaded: %s", current_grounder)
    if whisper_changed:
        transcriber = await asyncio.to_thread(
            WhisperTranscriber, model_size=current_whisper_model
        )
        logger.info("Transcriber reloaded with model=%s", current_whisper_model)
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
    """Capture and processing are decoupled into two tasks: one only writes raw
    audio to a per-session file (so the socket is always drained and never
    overflows/disconnects), the other transcribes + grounds it at its own pace.
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    audio_path = Path(CODA_BASE.join("sessions", session_id)) / "audio.pcm"
    if save_enabled and not save_files:
        open_save_files(current_language)

    # Register two coroutines in event loop as tasks:
    # `capture` writes audio, `process` transcribes it.
    capture = asyncio.create_task(write_audio_bytes(websocket, audio_path))
    process = asyncio.create_task(process_audio_bytes(websocket, audio_path))

    try:
        # Run the two tasks and gather whatever is surfaced from them. 
        # In our case we expect `capture` to surface `WebSocketDisconnect`
        # when the user ends recording session. 
        await asyncio.gather(capture, process)
    except WebSocketDisconnect:
        # `WebSocketDisconnect`` should surface from `capture` (the only task 
        # reading the socket).
        logger.info("WebSocket disconnected")
    finally:
        # `process` never raises on disconnect, so cancel it (and pending inference)
        # or it keeps looping after the session ends.
        process.cancel()
        # Similarly, tasks are created for pending_chunks in `process` that we
        # also have to cancel. 
        for task in pending_chunks.values():
            task.cancel()
        pending_chunks.clear()

        # `capture.cancel()`` is a safeguard for a DIFFERENT reason: on a non-disconnect
        # exit (e.g. `process` errored while still connected) `capture` may still be
        # running and reading the socket, so cancel it too to avoid an orphan.
        # Otherwise, it would naturally cancel when it surfaces the `WebSocketDisconnect`
        # exception.
        capture.cancel()

        # Wait for tasks to stop
        await asyncio.gather(capture, process, return_exceptions=True)

        # Drop the temporary audio buffer
        shutil.rmtree(audio_path.parent, ignore_errors=True)


async def write_audio_bytes(websocket: WebSocket, audio_path):
    """Capture: append each received audio message to the file. Never blocks."""
    with open(audio_path, "ab") as f:
        while True:
            f.write(await websocket.receive_bytes())
            f.flush()


async def process_audio_bytes(websocket: WebSocket, audio_path):
    """Process: tail the capture file, transcribe + ground each chunk at own pace."""
    processor = AudioProcessor()
    offset = 0
    while True:
        size = audio_path.stat().st_size if audio_path.exists() else 0
        if size <= offset:
            await asyncio.sleep(0.3)
            continue
        with open(audio_path, "rb") as f:
            f.seek(offset)
            data = f.read(size - offset)
        offset = size
        if not processor.add_audio(data):
            continue
        while True:
            result = processor.get_chunk()
            if result is None:
                break
            chunk_id, timestamp, chunk = result
            logger.info(f"Audio backlog: {datetime.now().timestamp() - timestamp:.1f}s")
            logger.info(f"Inference backlog: {len(pending_chunks)} chunks, "
                        f"{processor.chunk_duration} seconds each")

            # Backpressure: drop oldest chunk if too many pending
            if len(pending_chunks) >= MAX_PENDING_CHUNKS:
                oldest_id = next(iter(pending_chunks))
                pending_chunks[oldest_id].cancel()
                del pending_chunks[oldest_id]
                logger.warning(f"Dropped chunk {oldest_id} due to backpressure")
                await websocket.send_json({
                    "type": "warning",
                    "message": "Processing slower than audio - dropping old chunks"
                })

            # One bad chunk shouldn't kill the session
            try:
                await handle_chunk(websocket, chunk_id, timestamp, chunk)
            except Exception as e:
                logger.error(f"Error on chunk {chunk_id}: {e}", exc_info=True)


async def handle_chunk(websocket: WebSocket, chunk_id, timestamp, chunk):
    """Transcribe, ground, save, and start inference for one chunk."""
    # Transcribe audio to text
    original_transcript = None
    if (current_language != "en"
            and translation_mode == "whisper_translate"):
        # Whisper translates directly to English
        english_text = await transcriber.transcribe_audio(
            chunk, language=current_language, task="translate"
        )
    else:
        # Transcribe in the configured language
        transcript = await transcriber.transcribe_audio(
            chunk, language=current_language
        )
        english_text = transcript

        # If non-English with LLM mode, translate
        # (skip if transcript is too short to be real speech)
        if (current_language != "en"
                and translation_mode == "llm"
                and len(transcript.split()) > 1):
            original_transcript = transcript
            english_text = await translate_text(
                transcript, current_language
            )

    # Ground the (final, English) text without blocking the loop
    annotations = []
    if english_text:
        annotations = await asyncio.to_thread(
            grounder.annotate, english_text
        )

    if english_text:

        # Save transcripts and annotations if enabled
        if save_enabled:
            save_transcript(english_text, "en")
            if original_transcript and current_language != "en":
                save_transcript(original_transcript,
                                current_language)
            save_annotated_chunk(
                chunk_id, timestamp, english_text,
                annotations,
                original_text=original_transcript,
                original_language=(current_language
                                   if current_language != "en"
                                   else None),
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

        # Start inference in background (always on English text)
        inference_task = asyncio.create_task(
            process_inference(chunk_id, timestamp, english_text,
                            annotations, websocket)
        )
        pending_chunks[chunk_id] = inference_task


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
