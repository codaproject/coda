"""
Real-time Voice Transcription Server using OpenAI Whisper
Requirements:
    pip install fastapi uvicorn websockets whisper numpy scipy

To run:
    python server.py
"""

import asyncio
import logging
import os
from typing import Dict

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from coda.dialogue.whisper import WhisperTranscriber
from coda.dialogue import AudioProcessor
from coda.grounding.gilda_grounder import GildaGrounder

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
        logger.info(f"Inference result for {chunk_id}: {result['cod']}")

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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio streaming and transcription"""
    await websocket.accept()
    logger.info("WebSocket connection established")

    processor = AudioProcessor()

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

                    # Transcribe (now truly non-blocking)
                    transcript, annotations = await transcriber.transcribe_audio(chunk)

                    if transcript:
                        # Render annotations for display
                        annotations_rendered = render_annotations(annotations)

                        # Send transcript immediately
                        await websocket.send_json({
                            "type": "transcript",
                            "chunk_id": chunk_id,
                            "timestamp": timestamp,
                            "transcript": transcript,
                            "annotations": annotations_rendered
                        })
                        logger.info(f"Chunk {chunk_id}: {transcript}")

                        # Start inference in background
                        inference_task = asyncio.create_task(
                            process_inference(chunk_id, timestamp, transcript,
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
