import asyncio
import logging
import os

import torch
import whisper

from . import Transcriber

# For more info on models see
# https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages
DEFAULT_MODEL_SIZE = "small"

# Default threshold for filtering silent segments.
# Segments with no_speech_prob above this value are considered silence.
DEFAULT_NO_SPEECH_THRESHOLD = 0.6

logger = logging.getLogger(__name__)


class WhisperTranscriber(Transcriber):
    """Transcriber implementation using OpenAI's Whisper model."""
    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE,
                 no_speech_threshold: float = None):
        self.no_speech_threshold = (
            no_speech_threshold if no_speech_threshold is not None
            else DEFAULT_NO_SPEECH_THRESHOLD
        )
        self.device = self._resolve_device()
        # Whisper only supports fp16 on CUDA; CPU inference must use fp32.
        self.fp16 = self.device == "cuda"
        logger.info(f"Loading Whisper model: {model_size} (device={self.device})")
        self.model = whisper.load_model(model_size, device=self.device)
        logger.info("Whisper model loaded successfully")

    @staticmethod
    def _resolve_device() -> str:
        """Determine the torch device, honoring the CODA_DEVICE env var.

        Defaults to CPU. If "cuda" is requested but no CUDA device is
        available (e.g. a GPU image launched without --gpus), fall back to CPU.
        """
        requested = os.environ.get("CODA_DEVICE", "cpu").lower()
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CODA_DEVICE=cuda requested but no CUDA device is available; "
                "falling back to CPU"
            )
            return "cpu"
        return requested

    async def transcribe_file(self, file_path: str, language: str = "en",
                              task: str = "transcribe",
                              fp16: bool = False, verbose: bool = False):
        """Transcribe file asynchronously using thread pool."""
        return await asyncio.to_thread(
            self._sync_transcribe, file_path, language, task, fp16, verbose
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

    def _filter_segments(self, result: dict, language: str = "en") -> str:
        """Filter transcription segments based on no_speech_prob.

        Whisper tends to hallucinate phrases like "thank you for watching"
        during silence. This filters out segments where the model detects
        high probability of no speech.

        Only applied for English - Whisper's no_speech_prob is unreliable
        for other languages, often reporting high values on real speech.
        """
        segments = result.get("segments", [])
        if not segments:
            return result.get("text", "").strip()

        # Use a higher threshold for non-English since Whisper's
        # no_speech_prob tends to be inflated for other languages
        threshold = self.no_speech_threshold if language == "en" \
            else max(self.no_speech_threshold, 0.8)

        filtered_texts = []
        for segment in segments:
            no_speech_prob = segment.get("no_speech_prob", 0.0)
            if no_speech_prob < threshold:
                filtered_texts.append(segment.get("text", ""))
            else:
                logger.debug(
                    f"Filtered silent segment (no_speech_prob={no_speech_prob:.2f}): "
                    f"{segment.get('text', '')!r}"
                )

        return "".join(filtered_texts).strip()
