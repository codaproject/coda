import logging

import whisper

from . import Transcriber

# For more info on models see
# https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages
DEFAULT_MODEL_SIZE = "small"

logger = logging.getLogger(__name__)

class WhisperTranscriber(Transcriber):
    """Transcriber implementation using OpenAI's Whisper model."""
    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE):
        super().__init__()
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        logger.info("Whisper model loaded successfully")

    def transcribe_file(self, file_path: str, language: str = "en",
                              fp16: bool = False, verbose: bool = False):
        result = self.model.transcribe(
            file_path,
            language="en",  # Set to None for auto-detection
            fp16=False,
            verbose=False
        )
        return result
