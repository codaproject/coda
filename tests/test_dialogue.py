"""Tests for the transcriber, decoupled from grounding.

A fake transcribe_file lets us exercise transcribe_audio without loading a
real Whisper model or any network/grounder.
"""
import numpy as np

from coda.dialogue import Transcriber


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
