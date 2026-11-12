import asyncio
import contextlib
import logging
import math
import re
import string
import sys
import threading
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
# SimulStreaming logs its decoding context, which is transcript text
logging.getLogger("whisperlivekit.simul_whisper").setLevel(logging.WARNING)


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
        if text.startswith(prev):
            delta = text[len(prev):].strip()
        else:
            logger.warning("WhisperLiveKit revised the line at %s", start)
            delta = text
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


def _omit_empty_prompt(asr):
    """Decode a LocalAgreement ASR with no prompt when there is no prior text.

    The backends otherwise pass an empty prompt, which Whisper encodes as a
    prompt of one space.
    """
    transcribe = asr.transcribe

    def wrapper(audio, init_prompt=""):
        return transcribe(audio, init_prompt=init_prompt or None)

    asr.transcribe = wrapper


# A phrase said this many times in a row is a repetition loop
LOOP_COPIES = 4


def _bare(word):
    """Return a word without its case, spaces and surrounding punctuation."""
    return word.strip(string.punctuation + string.whitespace).lower()


def _loop_span(words):
    """Return (start, end, phrase length) of the shortest-phrase loop, or None.

    A loop is a phrase said at least LOOP_COPIES times in a row. Its span
    covers its whole copies and a last copy cut short by the end of words.
    """
    for n in range(1, len(words) // LOOP_COPIES + 1):
        run = 0
        for i in range(n, len(words)):
            run = run + 1 if words[i] == words[i - n] else 0
            if run >= n * (LOOP_COPIES - 1):
                start, end = i - run - n + 1, i + 1
                while words[end:end + n] == words[start:start + n]:
                    end += n
                # A last copy that the end of the decode cut short
                if words[end:] == words[start:start + len(words) - end]:
                    end = len(words)
                return start, end, n
    return None


def _drop_undecodable(words):
    """Return words without the characters Whisper could not decode.

    Tokens that are not valid text decode to U+FFFD, and a run of them is
    glued to one word. A word left with no letters is dropped, and the word
    before it ends where it ended, so committing it passes over its audio.
    """
    kept = []
    for word in words:
        if "\ufffd" in word.text:
            word.text = word.text.replace("\ufffd", "")
            if not _bare(word.text):
                if kept:
                    kept[-1].end = word.end
                continue
        kept.append(word)
    return kept


# Two or more characters repeated LOOP_COPIES times in a row within a word
INNER_LOOP = re.compile(r"(.{2,}?)\1{%d,}" % (LOOP_COPIES - 1))


def _collapse_inner(text):
    """Return a word with one copy of each run of characters it loops on.

    A last copy cut short by the end of the word is dropped too.
    """
    match = INNER_LOOP.search(text)
    if match is None:
        return text
    unit, rest = match[1], text[match.end():]
    if unit.startswith(rest):
        rest = ""
    return text[:match.start()] + unit + _collapse_inner(rest)


def _collapse_loops(asr):
    """Keep one copy of each phrase a LocalAgreement ASR decode loops on.

    Committed words prompt the next decode, so a committed loop keeps
    itself going. Collapsing before decodes are compared keeps it out of
    both. The copy kept ends where the last whole copy ended, so committing
    it passes over the loop's audio.
    """
    ts_words = asr.ts_words

    def wrapper(segments):
        words = _drop_undecodable(ts_words(segments))
        for word in words:
            word.text = _collapse_inner(word.text)
        while span := _loop_span([_bare(word.text) for word in words]):
            start, end, n = span
            copies = (end - start) // n
            logger.debug("Collapsed a phrase repeated %d times", copies)
            words[start + n - 1].end = words[start + n * copies - 1].end
            words = words[:start + n] + words[end:]
        return words

    asr.ts_words = wrapper


DTW_LOCK = threading.Lock()


def _serialize_dtw():
    """Run the loaded Whisper word-timing DTW in one thread at a time.

    The DTW is a Numba function compiled with ``parallel=True``, and Numba's
    default threading layer aborts the process when two threads run such a
    function at once.
    """
    for name in ("mlx_whisper.timing", "whisperlivekit.whisper.timing"):
        timing = sys.modules.get(name)
        if timing is None or getattr(timing.dtw_cpu, "serialized", False):
            continue

        def serialized(x, dtw=timing.dtw_cpu):
            with DTW_LOCK:
                return dtw(x)

        serialized.serialized = True
        timing.dtw_cpu = serialized


class WhisperLiveKitTranscriber(StreamingTranscriber):
    """Streaming backend running WhisperLiveKit's engine in-process.

    Feeds raw PCM (s16le, 16 kHz, what the browser already sends) to a
    per-connection WhisperLiveKit AudioProcessor and turns its incremental
    responses into committed and preview TranscriptEvents. With the default
    faster-whisper backend it reuses the same faster-whisper model as the
    `faster-whisper` backend; the engine (model) is loaded once and shared.
    """
    MODELS = ("tiny", "base", "small", "medium",
              "large", "large-v2", "large-v3", "large-v3-turbo",
              *FINETUNED_MODELS)
    DEFAULT_MODEL = DEFAULT_MODEL_SIZE
    LANGUAGES = {**get_whisper_languages(), **FINETUNED_LANGUAGES}
    LANGUAGE_AT_BUILD = True

    @classmethod
    def create(cls, model=None, language=None):
        return cls(model_size=model or cls.DEFAULT_MODEL, language=language)

    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE, language=None):
        from whisperlivekit import TranscriptionEngine
        from coda.config import settings
        wlk = settings.dialogue.whisper_livekit
        # The language is fixed when the engine is built, so this backend
        # ignores the per-stream language argument
        lan, condition = decode_options(
            model_size, language or settings.dialogue.language)
        if model_size in FINETUNED_MODELS:
            mlx = wlk.backend == "mlx-whisper"
            model = {"model_dir": get_model_path(model_size, mlx=mlx)}
        else:
            model = {"model_size": model_size}
        # The engine is a singleton that ignores the arguments of later
        # constructions, so building for another model or language needs a
        # reset. Streams already running keep the engine they were given.
        TranscriptionEngine.reset()
        self._engine = TranscriptionEngine(
            **model,
            lan=lan,
            backend=wlk.backend,
            backend_policy=wlk.policy,
            pcm_input=True,
        )
        if not condition and wlk.policy == "localagreement":
            _disable_text_conditioning(self._engine.asr)
        if wlk.policy == "localagreement":
            _omit_empty_prompt(self._engine.asr)
            _collapse_loops(self._engine.asr)
        _serialize_dtw()

    async def stream(self, audio: AsyncIterator[bytes], *,
                     language: str = "en",
                     task: str = "transcribe") -> AsyncIterator[TranscriptEvent]:
        from whisperlivekit import AudioProcessor
        processor = AudioProcessor(transcription_engine=self._engine)
        # Keep every committed word, so each line only grows under the same
        # start. Dropping old words would re-key a long line and resend it.
        processor.tokens_alignment._retention_seconds = math.inf
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
