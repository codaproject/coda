import asyncio
import logging

import whisper

from . import Transcriber
from coda.grounding import BaseGrounder

# For more info on models see
# https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages
DEFAULT_MODEL_SIZE = "small"

logger = logging.getLogger(__name__)

class WhisperTranscriber(Transcriber):
    """Transcriber implementation using OpenAI's Whisper model."""
    def __init__(self, grounder: BaseGrounder, model_size: str = DEFAULT_MODEL_SIZE):
        super().__init__(grounder=grounder)
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        logger.info("Whisper model loaded successfully")

    async def transcribe_file(self, file_path: str, language: str = "en",
                              fp16: bool = False, verbose: bool = False):
        """Transcribe file asynchronously using thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._sync_transcribe,
            file_path, language, fp16, verbose
        )

    def _sync_transcribe(self, file_path: str, language: str,
                        fp16: bool, verbose: bool):
        """Synchronous transcription method."""
        return self.model.transcribe(
            file_path,
            language=language,
            fp16=fp16,
            verbose=verbose
        )
