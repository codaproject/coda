"""
Real-time Voice Transcription Server using OpenAI Whisper
Requirements:
    pip install fastapi uvicorn websockets whisper numpy scipy
    
To run:
    python server.py
"""

import logging

import os
import gilda
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from coda.dialogue.whisper import WhisperTranscriber
from coda.dialogue import AudioProcessor

app = FastAPI()
transcriber = WhisperTranscriber(model_size="medium")

logger = logging.getLogger(__name__)

here = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(here, "templates")


def render_annotations(annotations):
    """Render annotations as a string."""
    parts = []
    for ann in annotations:
        term = ann.matches[0].term
        curie = term.get_curie()
        name = term.entry_name
        text = ann.text
        part = f"{text} = {curie} ({name})"
        parts.append(part)
    return "\n".join(parts)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio streaming and transcription"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    processor = AudioProcessor()
    
    try:
        while True:
            # Receive audio data
            audio_bytes = await websocket.receive_bytes()
            
            # Add to buffer and check if ready for processing
            if processor.add_audio(audio_bytes):
                # Get chunk for processing
                chunk = processor.get_chunk()
                if chunk is not None:
                    # Transcribe in background
                    transcript, annotations = await transcriber.transcribe_audio(chunk)

                    annotations_str = render_annotations(annotations)
                    
                    if transcript:
                        # Send transcript back to client
                        await websocket.send_json({
                            "transcript": transcript,
                            "annotations": annotations_str
                        })
                        logger.info(f"Transcribed: {transcript}, Annotations: {annotations_str}")
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        processor.clear_buffer()
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
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
