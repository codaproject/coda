"""Tests for the transcriber, decoupled from grounding.

A fake transcribe_file lets us exercise transcribe_audio without loading a
real Whisper model or any network/grounder.
"""
import asyncio
import time

import numpy as np
import pytest

from coda.dialogue import (
    ChunkedTranscriber,
    TRANSCRIBER_BACKENDS,
    TranscriptEvent,
)
from coda.dialogue.faster_whisper import FasterWhisperTranscriber
from coda.dialogue.whisper import WhisperTranscriber
from coda.dialogue.indic_conformer import IndicConformerTranscriber
from coda.dialogue.whisper_livekit import (
    WhisperLiveKitTranscriber,
    _events_from_response,
)
from coda.config import settings


class _FakeTranscriber(ChunkedTranscriber):
    async def transcribe_file(self, file_path, language="en", task="transcribe",
                              fp16=False, verbose=False):
        return {"text": "patient had a fever", "segments": []}


async def test_transcribe_audio_returns_text():
    transcriber = _FakeTranscriber()
    audio = np.full(16000, 1000, dtype=np.int16)  # non-silent
    text = await transcriber.transcribe_audio(audio)
    assert isinstance(text, str)
    assert text == "patient had a fever"


async def test_transcribe_audio_silent_returns_empty_string():
    transcriber = _FakeTranscriber()
    audio = np.zeros(16000, dtype=np.int16)  # silent -> early return
    text = await transcriber.transcribe_audio(audio)
    assert text == ""


async def test_chunked_stream_yields_committed_event():
    transcriber = _FakeTranscriber()
    # One 3-second chunk of non-silent audio (16 kHz * 3s = 48000 int16 samples)
    pcm = np.full(48000, 1000, dtype=np.int16).tobytes()

    async def audio():
        yield pcm

    events = [ev async for ev in transcriber.stream(audio())]
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, TranscriptEvent)
    assert ev.committed is True
    assert ev.text == "patient had a fever"
    assert isinstance(ev.id, str) and ev.id
    assert isinstance(ev.timestamp, float)


def test_faster_whisper_backend_registered():
    assert "faster-whisper" in TRANSCRIBER_BACKENDS


def _faster_whisper_without_model():
    # Bypass __init__ to avoid downloading/loading a real model.
    transcriber = object.__new__(FasterWhisperTranscriber)
    transcriber.no_speech_threshold = 0.6
    return transcriber


def test_faster_whisper_filter_drops_silent_segments_english():
    transcriber = _faster_whisper_without_model()
    result = {
        "text": "ignored",
        "segments": [
            {"text": "patient had a fever", "no_speech_prob": 0.1},
            {"text": " thank you for watching", "no_speech_prob": 0.9},
        ],
    }
    assert transcriber._filter_segments(result, language="en") == \
        "patient had a fever"


def test_faster_whisper_filter_uses_higher_threshold_non_english():
    transcriber = _faster_whisper_without_model()
    # no_speech_prob 0.7 exceeds the English threshold but not the 0.8
    # non-English floor, so the segment is kept.
    result = {
        "text": "ignored",
        "segments": [{"text": "জ্বর ছিল", "no_speech_prob": 0.7}],
    }
    assert transcriber._filter_segments(result, language="bn") == "জ্বর ছিল"


def test_indic_conformer_backend_registered():
    assert "indic-conformer" in TRANSCRIBER_BACKENDS


def test_indic_conformer_rejects_unknown_decoding():
    with pytest.raises(ValueError):
        IndicConformerTranscriber(decoding="beam")


def test_indic_conformer_decoding_strategies():
    assert IndicConformerTranscriber.DEFAULT_MODEL in \
        IndicConformerTranscriber.MODELS
    assert IndicConformerTranscriber.normalize_language("bn") == "bn"
    assert IndicConformerTranscriber.normalize_language("en") is None


def _indic_conformer_without_model(decoding="ctc"):
    # Bypass __init__ to avoid downloading a real model.
    transcriber = object.__new__(IndicConformerTranscriber)
    transcriber.decoding = decoding
    transcriber.model = lambda wav, language, dec: "জ্বর ছিল"
    return transcriber


async def test_indic_conformer_transcribes_audio():
    transcriber = _indic_conformer_without_model()
    audio = np.full(16000, 1000, dtype=np.int16)
    assert await transcriber.transcribe_audio(audio, language="bn") == "জ্বর ছিল"


async def test_indic_conformer_unsupported_language_returns_empty_string():
    transcriber = _indic_conformer_without_model()
    audio = np.full(16000, 1000, dtype=np.int16)
    assert await transcriber.transcribe_audio(audio, language="en") == ""


async def test_indic_conformer_rejects_wrong_sample_rate():
    transcriber = _indic_conformer_without_model()
    audio = np.full(16000, 1000, dtype=np.int16)
    assert await transcriber.transcribe_audio(
        audio, sample_rate=8000, language="bn") == ""


async def test_indic_conformer_does_not_block_the_event_loop():
    transcriber = _indic_conformer_without_model()
    transcriber.model = lambda wav, language, dec: time.sleep(0.3) or "জ্বর"
    audio = np.full(16000, 1000, dtype=np.int16)
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    await transcriber.transcribe_audio(audio, language="bn")
    beat.cancel()
    assert ticks > 5


def test_whisper_livekit_backend_registered():
    assert "whisper-livekit" in TRANSCRIBER_BACKENDS


def test_whisper_livekit_runtimes():
    assert "mlx-whisper" in WhisperLiveKitTranscriber.RUNTIMES
    assert WhisperLiveKitTranscriber.default_runtime() == \
        settings.dialogue.whisper_livekit.backend


def test_turbo_offered_by_every_whisper_backend():
    assert "large-v3-turbo" in WhisperTranscriber.MODELS
    assert "large-v3-turbo" in FasterWhisperTranscriber.MODELS
    assert "large-v3-turbo" in WhisperLiveKitTranscriber.MODELS


def test_backends_without_a_runtime_choice():
    assert FasterWhisperTranscriber.RUNTIMES == ()
    assert FasterWhisperTranscriber.default_runtime() is None


def test_whisper_livekit_events_from_response():
    # A line's text is extended across responses (and the first line repeats);
    # only the new suffix per line should be emitted, plus changed previews.
    # start values are HMS strings, matching the real server.
    responses = [
        {"lines": [{"text": "hello there", "start": "0:00:00.00",
                    "end": "0:00:01.00"}],
         "buffer_transcription": "how are"},
        {"lines": [{"text": "hello there how are you", "start": "0:00:00.00",
                    "end": "0:00:02.00"}],
         "buffer_transcription": "today"},
        {"lines": [{"text": "hello there how are you", "start": "0:00:00.00",
                    "end": "0:00:02.00"},
                   {"text": "fine thanks", "start": "0:00:02.00",
                    "end": "0:00:03.00"}],
         "buffer_transcription": ""},
    ]
    state = {"emitted": {}, "preview": ""}
    events = []
    for msg in responses:
        events.extend(_events_from_response(msg, state))

    committed = [e.text for e in events if e.committed]
    previews = [e.text for e in events if not e.committed]
    assert committed == ["hello there", "how are you", "fine thanks"]
    assert previews == ["how are", "today"]
    assert all(e.committed in (True, False) for e in events)
