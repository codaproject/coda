"""Tests for the transcriber, decoupled from grounding.

A fake transcribe_file lets us exercise transcribe_audio without loading a
real Whisper model or any network/grounder.
"""
import numpy as np

from coda.dialogue import TRANSCRIBER_BACKENDS, Transcriber
from coda.dialogue.faster_whisper import FasterWhisperTranscriber


class _FakeTranscriber(Transcriber):
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
