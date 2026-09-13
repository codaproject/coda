__all__ = ["Translator", "TRANSLATOR_BACKENDS", "create_translator",
           "get_asr_task"]

import logging

logger = logging.getLogger(__name__)

# Selectable translation backends, chosen via the app's translation_mode
# setting. whisper_translate is carried out by the transcriber itself, llm
# and ctranslate2 translate the finished transcript as a second step.
TRANSLATOR_BACKENDS = ("whisper_translate", "llm", "ctranslate2")


def _load_backend_class(backend: str):
    """Import and return the Translator subclass for `backend`.

    Only the requested backend is imported, so a deployment needn't install
    every backend's dependencies. Raises ImportError if its deps are missing.
    """
    if backend == "whisper_translate":
        from .whisper_direct import WhisperDirectTranslator
        return WhisperDirectTranslator
    if backend == "llm":
        from .llm import LlmTranslator
        return LlmTranslator
    if backend == "ctranslate2":
        from .ct2 import CTranslate2Translator
        return CTranslate2Translator
    raise ValueError(
        f"Unknown translator backend {backend!r}; "
        f"choose from {TRANSLATOR_BACKENDS}"
    )


def create_translator(backend: str, **kwargs):
    """Build a Translator for the named backend."""
    return _load_backend_class(backend).create(**kwargs)


def get_asr_task(backend: str) -> str:
    """Return the transcription task a backend needs the transcriber to run.

    Backends that translate speech directly ask for "translate", the rest leave
    the transcriber in its normal "transcribe" mode.
    """
    return _load_backend_class(backend).ASR_TASK


class Translator:
    """Abstract translation backend.

    Turns transcript text into English. A backend that instead translates
    during transcription declares ASR_TASK and passes text through unchanged.
    """
    # Transcription task to run when this backend is selected
    ASR_TASK = "transcribe"

    @classmethod
    def create(cls, **kwargs):
        """Instantiate this backend."""
        raise NotImplementedError

    async def translate(self, text: str, source_language: str,
                        language_name: str = None) -> str:
        raise NotImplementedError
