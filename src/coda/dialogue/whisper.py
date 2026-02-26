import asyncio
import logging

import whisper

from . import Transcriber
from coda.grounding import BaseGrounder

# For more info on models see
# https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages
DEFAULT_MODEL_SIZE = "small"

# Default threshold for filtering silent segments.
# Segments with no_speech_prob above this value are considered silence.
DEFAULT_NO_SPEECH_THRESHOLD = 0.6

logger = logging.getLogger(__name__)


class WhisperTranscriber(Transcriber):
    """Transcriber implementation using OpenAI's Whisper model."""
    def __init__(self, grounder: BaseGrounder, model_size: str = DEFAULT_MODEL_SIZE,
                 no_speech_threshold: float = None):
        super().__init__(grounder=grounder)
        self.no_speech_threshold = (
            no_speech_threshold if no_speech_threshold is not None
            else DEFAULT_NO_SPEECH_THRESHOLD
        )
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        logger.info("Whisper model loaded successfully")

    async def transcribe_file(self, file_path: str, language: str = "en",
                              task: str = "transcribe",
                              fp16: bool = False, verbose: bool = False):
        """Transcribe file asynchronously using thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._sync_transcribe,
            file_path, language, task, fp16, verbose
        )

    def _sync_transcribe(self, file_path: str, language: str,
                        task: str, fp16: bool, verbose: bool):
        """Synchronous transcription method."""
        return self.model.transcribe(
            file_path,
            language=language,
            task=task,
            fp16=fp16,
            verbose=verbose
        )

    def _filter_segments(self, result: dict) -> str:
        """Filter transcription segments based on no_speech_prob.

        Whisper tends to hallucinate phrases like "thank you for watching"
        during silence. This filters out segments where the model detects
        high probability of no speech.

        Parameters
        ----------
        result :
            The result dictionary from Whisper's transcribe() method

        Returns
        -------
        str
            Filtered transcription text with silent segments removed
        """
        segments = result.get("segments", [])
        if not segments:
            return result.get("text", "").strip()

        filtered_texts = []
        for segment in segments:
            no_speech_prob = segment.get("no_speech_prob", 0.0)
            if no_speech_prob < self.no_speech_threshold:
                filtered_texts.append(segment.get("text", ""))
            else:
                logger.debug(
                    f"Filtered silent segment (no_speech_prob={no_speech_prob:.2f}): "
                    f"{segment.get('text', '')!r}"
                )

        return "".join(filtered_texts).strip()
