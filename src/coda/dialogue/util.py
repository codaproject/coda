"""Shared language metadata for Whisper-based transcribers."""
from functools import cache


@cache
def get_whisper_languages():
    """Share Whisper's language names across its transcription backends."""
    from whisper.tokenizer import LANGUAGES

    return {code: name.title() for code, name in LANGUAGES.items()}
