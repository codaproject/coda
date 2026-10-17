import asyncio
import contextlib
import inspect
import logging
import time
import uuid
from typing import AsyncIterator

from . import StreamingTranscriber, TranscriptEvent
from .finetuned_whisper import FINETUNED_LANGUAGES, FINETUNED_MODELS, \
    decode_options, get_model_path
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


def _disable_text_conditioning(asr):
    """Decode each buffer of a LocalAgreement ASR without its previous text.

    The backends always prompt with the committed text and condition on it,
    which sends some fine-tunes into repetition loops. The faster-whisper
    backend holds a WhisperModel, the MLX one a transcribe function.
    """
    def unconditioned(decode):
        def wrapper(audio, **kwargs):
            kwargs.update(initial_prompt=None, condition_on_previous_text=False)
            return decode(audio, **kwargs)
        return wrapper

    if hasattr(asr.model, "transcribe"):
        asr.model.transcribe = unconditioned(asr.model.transcribe)
    else:
        asr.model = unconditioned(asr.model)


class WhisperLiveKitTranscriber(StreamingTranscriber):
    """Streaming backend running WhisperLiveKit's engine in-process.

    Feeds raw PCM (s16le, 16 kHz, what the browser already sends) to a
    per-connection WhisperLiveKit AudioProcessor and turns its incremental
    responses into committed and preview TranscriptEvents. With the default
    faster-whisper backend it reuses the same faster-whisper model as the
    `faster-whisper` backend; the engine (model) is loaded once and shared.
    """
    MODELS = ("tiny", "base", "small", "medium",
              "large", "large-v2", "large-v3", *FINETUNED_MODELS)
    DEFAULT_MODEL = DEFAULT_MODEL_SIZE
    LANGUAGES = {**get_whisper_languages(), **FINETUNED_LANGUAGES}

    @classmethod
    def create(cls, model=None):
        return cls(model_size=model or cls.DEFAULT_MODEL)

    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE):
        from whisperlivekit import TranscriptionEngine
        from coda.config import settings
        wlk = settings.dialogue.whisper_livekit
        # The language is fixed when the engine is built, so this backend
        # transcribes whatever dialogue.language was set to at startup and
        # ignores the per-stream language argument.
        lan, condition = decode_options(model_size, settings.dialogue.language)
        if model_size in FINETUNED_MODELS:
            mlx = wlk.backend == "mlx-whisper"
            model = {"model_dir": get_model_path(model_size, mlx=mlx)}
        else:
            model = {"model_size": model_size}
        self._engine = TranscriptionEngine(
            **model,
            lan=lan,
            backend=wlk.backend,
            backend_policy=wlk.policy,
            vac=wlk.vac,
            pcm_input=True,
        )
        if not condition and wlk.policy == "localagreement":
            _disable_text_conditioning(self._engine.asr)
        # faster-whisper's own silence/noise suppression thresholds (see
        # decode_options in settings.yaml). WLK unpacks the backend's
        # transcribe_kargs dict into every faster-whisper transcribe() call,
        # and the backend is shared by all connections, so filling that dict
        # once here applies process-wide. Only the faster-whisper backend
        # takes these option names.
        self.decode_options = dict(wlk.decode_options)
        if self.decode_options and wlk.backend == "faster-whisper":
            # faster-whisper rejects an unknown option only when it decodes,
            # inside WLK's worker, where the error becomes a logged warning and
            # an empty transcript. Check the option names here instead.
            from faster_whisper import WhisperModel
            accepted = inspect.signature(WhisperModel.transcribe).parameters
            unknown = set(self.decode_options) - set(accepted)
            if unknown:
                raise ValueError(f"Unknown decode_options {sorted(unknown)}")
            self._engine.asr.transcribe_kargs.update(self.decode_options)
            logger.info("faster-whisper decode options: %s", self.decode_options)

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
