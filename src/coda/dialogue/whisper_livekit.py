import asyncio
import contextlib
import logging
import os
import time
import uuid
from typing import AsyncIterator

from . import StreamingTranscriber, TranscriptEvent

# Model size used when none is supplied (matches the other backends).
DEFAULT_MODEL_SIZE = "small"

logger = logging.getLogger(__name__)


def _events_from_response(msg: dict, state: dict):
    """Yield TranscriptEvents for one WhisperLiveKit response.

    WLK extends a line's text as it commits clauses (lines are keyed by a stable
    `start`; silence lines have empty text), and each response repeats the
    current window. Emit only the newly-appended suffix per line as a committed
    event, plus the interim `buffer_transcription` as a preview. `state` carries
    `emitted` (start -> text already emitted) and `preview` across responses.
    """
    emitted = state["emitted"]
    for line in msg.get("lines", []):
        text = (line.get("text") or "").strip()
        if not text:
            continue
        start = line.get("start")
        prev = emitted.get(start, "")
        if text == prev:
            continue
        delta = text[len(prev):].strip() if text.startswith(prev) else text
        emitted[start] = text
        if delta:
            yield TranscriptEvent(id=uuid.uuid4().hex, timestamp=time.time(),
                                  text=delta, committed=True,
                                  speaker=line.get("speaker"))
    buf = (msg.get("buffer_transcription") or "").strip()
    if buf and buf != state["preview"]:
        yield TranscriptEvent(id=uuid.uuid4().hex, timestamp=time.time(),
                              text=buf, committed=False)
    state["preview"] = buf


class WhisperLiveKitTranscriber(StreamingTranscriber):
    """Streaming backend running WhisperLiveKit's engine in-process.

    Feeds raw PCM (s16le, 16 kHz, what the browser already sends) to a
    per-connection WhisperLiveKit AudioProcessor and turns its incremental
    responses into committed and preview TranscriptEvents. With the default
    faster-whisper backend it reuses the same faster-whisper model as the
    `faster-whisper` backend; the engine (model) is loaded once and shared.
    """
    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE):
        from whisperlivekit import TranscriptionEngine
        self._engine = TranscriptionEngine(
            model_size=model_size,
            lan=os.environ.get("CODA_WLK_LANGUAGE", "en"),
            backend=os.environ.get("CODA_WLK_BACKEND", "faster-whisper"),
            backend_policy=os.environ.get("CODA_WLK_POLICY", "localagreement"),
            pcm_input=True,
        )

    async def stream(self, audio: AsyncIterator[bytes], *,
                     language: str = "en",
                     task: str = "transcribe") -> AsyncIterator[TranscriptEvent]:
        from whisperlivekit import AudioProcessor
        processor = AudioProcessor(transcription_engine=self._engine)
        results = await processor.create_tasks()
        forwarder = asyncio.create_task(self._forward(processor, audio))
        state = {"emitted": {}, "preview": ""}
        try:
            async for response in results:
                msg = response.to_dict() if hasattr(response, "to_dict") \
                    else response
                for event in _events_from_response(msg, state):
                    yield event
        finally:
            forwarder.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await forwarder
            await processor.cleanup()

    async def _forward(self, processor, audio: AsyncIterator[bytes]):
        """Feed incoming audio to the processor, then signal end-of-audio."""
        async for data in audio:
            await processor.process_audio(data)
        await processor.process_audio(b"")
