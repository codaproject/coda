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
from coda.llm_api import create_llm_client

app = FastAPI()
transcriber = WhisperTranscriber(grounder=GildaGrounder(),
                                 model_size="medium")

# HTTP client for inference agent
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:5123")
inference_client = httpx.AsyncClient(base_url=INFERENCE_URL, timeout=30.0)

# Queue management for backpressure
MAX_PENDING_CHUNKS = 3
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
current_llm_provider = "openai"
current_llm_model = "gpt-4o-mini"
# "whisper_translate" = use whisper task="translate" (direct speech-to-English)
# "llm" = transcribe in original language, then translate via LLM
translation_mode = "whisper_translate"


class SettingsRequest(BaseModel):
    language: Optional[str] = None
    save_enabled: Optional[bool] = None
    whisper_model: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    translation_mode: Optional[str] = None


def get_language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


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
        loop = asyncio.get_running_loop()
        translation = await loop.run_in_executor(None, llm.call, prompt)
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
        await websocket.send_json({
            "type": "inference",
            **result
        })
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
        await websocket.send_json({
            "type": "error",
            "chunk_id": chunk_id,
            "error": "Inference timeout"
        })
    except httpx.ConnectError:
        logger.error(f"Cannot connect to inference agent for chunk {chunk_id}")
        await websocket.send_json({
            "type": "error",
            "chunk_id": chunk_id,
            "error": "Inference agent unavailable"
        })
    except Exception as e:
        logger.error(f"Inference error for chunk {chunk_id}: {e}", exc_info=True)
        await websocket.send_json({
            "type": "error",
            "chunk_id": chunk_id,
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
        "llm_provider": current_llm_provider,
        "llm_model": current_llm_model,
        "translation_mode": translation_mode,
    }


@app.post("/settings")
async def update_settings(req: SettingsRequest):
    """Update server settings."""
    global current_language, save_enabled, transcriber
    global current_whisper_model, current_llm_provider, current_llm_model
    global translation_mode
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
    if req.whisper_model is not None and req.whisper_model != current_whisper_model:
        current_whisper_model = req.whisper_model
        logger.info(f"Reloading Whisper model: {current_whisper_model}")
        # Load in a thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        new_transcriber = await loop.run_in_executor(
            None,
            lambda: WhisperTranscriber(grounder=GildaGrounder(),
                                       model_size=current_whisper_model)
        )
        transcriber = new_transcriber
        logger.info(f"Whisper model reloaded: {current_whisper_model}")
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
        "llm_provider": current_llm_provider,
        "llm_model": current_llm_model,
        "translation_mode": translation_mode,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio streaming and transcription"""
    await websocket.accept()
    logger.info("WebSocket connection established")

    processor = AudioProcessor()

    # Open save files at the start of a recording session if saving is enabled
    if save_enabled and not save_files:
        paths = open_save_files(current_language)
        logger.info(f"Opened save files for session: {paths}")

    try:
        while True:
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

            # Receive audio data
            audio_bytes = await websocket.receive_bytes()

            # Add to buffer and check if ready for processing
            if processor.add_audio(audio_bytes):
                # Get chunk with ID and timestamp
                result = processor.get_chunk()
                if result is not None:
                    chunk_id, timestamp, chunk = result

                    # Transcribe audio
                    original_transcript = None
                    if (current_language != "en"
                            and translation_mode == "whisper_translate"):
                        # Whisper translates directly to English
                        transcript, annotations = (
                            await transcriber.transcribe_audio(
                                chunk, language=current_language,
                                task="translate"
                            )
                        )
                        english_text = transcript
                    else:
                        # Transcribe in the configured language
                        transcript, annotations = (
                            await transcriber.transcribe_audio(
                                chunk, language=current_language
                            )
                        )
                        english_text = transcript

                        # If non-English with LLM mode, translate
                        if (current_language != "en"
                                and translation_mode == "llm"):
                            original_transcript = transcript
                            english_text = await translate_text(
                                transcript, current_language
                            )
                            # Re-ground on the English translation
                            annotations = transcriber.grounder.annotate(
                                english_text
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

                        # Render annotations for display
                        annotations_rendered = render_annotations(annotations)

                        # Send transcript to client
                        msg = {
                            "type": "transcript",
                            "chunk_id": chunk_id,
                            "timestamp": timestamp,
                            "transcript": english_text,
                            "annotations": annotations_rendered,
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

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        # Cancel all pending inference tasks
        for task in pending_chunks.values():
            task.cancel()
        pending_chunks.clear()
        processor.clear_buffer()

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass
        processor.clear_buffer()


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
