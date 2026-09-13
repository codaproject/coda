"""Translation of transcript text through a general-purpose LLM."""
import asyncio
import logging

from coda.llm_api import create_llm_client

from . import Translator

logger = logging.getLogger(__name__)


class LlmTranslator(Translator):
    """Translates a transcript to English with the configured LLM.

    The client is built per call so provider and model changes take effect
    without restarting an interview.
    """

    @classmethod
    def create(cls, provider=None, model=None, **kwargs):
        return cls(provider=provider, model=model)

    def __init__(self, provider=None, model=None):
        self.provider = provider
        self.model = model

    async def translate(self, text, source_language, language_name=None):
        lang_name = language_name or source_language
        prompt = (f"Translate the following {lang_name} text to English. "
                  f"Return only the translation, nothing else.\n\n{text}")
        try:
            llm = create_llm_client(provider=self.provider, model=self.model)
            translation = await asyncio.to_thread(llm.call, prompt)
            return translation.strip()
        except Exception as e:
            logger.error(f"Translation error: {e}")
            # Fall back to the untranslated text
            return text
