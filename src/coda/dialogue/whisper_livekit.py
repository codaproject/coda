import asyncio
import contextlib
import logging
import time
import uuid
from typing import AsyncIterator

from . import StreamingTranscriber, TranscriptEvent
from .util import get_whisper_languages

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
    MODELS = ("tiny", "base", "small", "medium",
              "large", "large-v2", "large-v3")
    DEFAULT_MODEL = DEFAULT_MODEL_SIZE
    LANGUAGES = get_whisper_languages()

    @classmethod
    def create(cls, model=None):
        return cls(model_size=model or cls.DEFAULT_MODEL)

    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE):
        from coda.config import settings
        self._model_size = model_size
        self._settings = settings.dialogue.whisper_livekit
        self._language = settings.dialogue.language
        self._engine = self._build_engine(self._language)

    def _build_engine(self, language: str):
        from whisperlivekit import TranscriptionEngine
        # TranscriptionEngine is a singleton whose __init__ short-circuits once
        # initialized, and its language is fixed at construction. Resetting is
        # the only way to load a different one.
        TranscriptionEngine.reset()
        return TranscriptionEngine(
            model_size=self._model_size,
            lan=language,
            backend=self._settings.backend,
            backend_policy=self._settings.policy,
            confidence_validation=self._settings.confidence_validation,
            buffer_trimming_sec=self._settings.buffer_trimming_sec,
            pcm_input=True,
        )

    def _engine_for(self, language: str):
        """Return an engine transcribing `language`, rebuilding if it changed.

        The language is fixed when the engine is constructed, so a different
        one means reloading the model. That only happens when the interview
        language actually changes.
        """
        if language and language != self._language:
            logger.info("Reloading whisper-livekit engine for language %r",
                        language)
            self._engine = self._build_engine(language)
            self._language = language
        return self._engine

    async def stream(self, audio: AsyncIterator[bytes], *,
                     language: str = "en",
                     task: str = "transcribe") -> AsyncIterator[TranscriptEvent]:
        from whisperlivekit import AudioProcessor
        processor = AudioProcessor(
            transcription_engine=self._engine_for(language))
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
