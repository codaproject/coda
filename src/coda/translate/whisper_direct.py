"""Translation carried out by the transcriber rather than as a second step."""
from . import Translator


class WhisperDirectTranslator(Translator):
    """Relies on Whisper's speech-to-English translation task.

    The transcript already arrives in English, so translate() returns it
    unchanged. Only the whisper backend offers this task, other transcribers
    need the transcript translated separately.
    """

    ASR_TASK = "translate"

    @classmethod
    def create(cls, **kwargs):
        return cls()

    async def translate(self, text, source_language, language_name=None):
        return text
