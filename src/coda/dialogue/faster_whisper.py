import asyncio
import logging
import os

import torch
from faster_whisper import WhisperModel

from . import Transcriber

# For available model sizes see
# https://github.com/SYSTRAN/faster-whisper#usage
DEFAULT_MODEL_SIZE = "small"

# Default threshold for filtering silent segments.
# Segments with no_speech_prob above this value are considered silence.
DEFAULT_NO_SPEECH_THRESHOLD = 0.6

logger = logging.getLogger(__name__)


class FasterWhisperTranscriber(Transcriber):
    """Transcriber using the CTranslate2-based faster-whisper runtime.

    Same Whisper weights as WhisperTranscriber but with a faster, lower-memory
    backend and a built-in Silero VAD filter for suppressing silence.
    """
    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE,
                 no_speech_threshold: float = None,
                 compute_type: str = None,
                 vad_filter: bool = True):
        self.no_speech_threshold = (
            no_speech_threshold if no_speech_threshold is not None
            else DEFAULT_NO_SPEECH_THRESHOLD
        )
        self.vad_filter = vad_filter
        self.device = self._resolve_device()
        self.compute_type = compute_type or self._resolve_compute_type()
        logger.info(
            f"Loading faster-whisper model: {model_size} "
            f"(device={self.device}, compute_type={self.compute_type})"
        )
        self.model = WhisperModel(
            model_size, device=self.device, compute_type=self.compute_type
        )
        logger.info("faster-whisper model loaded successfully")

    @staticmethod
    def _resolve_device() -> str:
        """Determine the device, honoring the CODA_DEVICE env var.

        Defaults to CPU. If "cuda" is requested but no CUDA device is
        available, fall back to CPU.
        """
        requested = os.environ.get("CODA_DEVICE", "cpu").lower()
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CODA_DEVICE=cuda requested but no CUDA device is available; "
                "falling back to CPU"
            )
            return "cpu"
        return requested

    def _resolve_compute_type(self) -> str:
        """Pick a compute type, honoring CODA_FASTER_WHISPER_COMPUTE_TYPE.

        Defaults to int8 on CPU and float16 on CUDA.
        """
        override = os.environ.get("CODA_FASTER_WHISPER_COMPUTE_TYPE", "").strip()
        if override:
            return override
        return "float16" if self.device == "cuda" else "int8"

    async def transcribe_file(self, file_path: str, language: str = "en",
                              task: str = "transcribe",
                              fp16: bool = False, verbose: bool = False):
        """Transcribe file asynchronously using thread pool."""
        return await asyncio.to_thread(
            self._sync_transcribe, file_path, language, task
        )

    def _sync_transcribe(self, file_path: str, language: str, task: str):
        """Synchronous transcription returning a whisper-compatible dict."""
        segments, _info = self.model.transcribe(
            file_path,
            language=language,
            task=task,
            vad_filter=self.vad_filter,
        )
        seg_dicts = [
            {
                "text": seg.text,
                "no_speech_prob": seg.no_speech_prob,
                "start": seg.start,
                "end": seg.end,
            }
            for seg in segments
        ]
        text = "".join(seg["text"] for seg in seg_dicts).strip()
        return {"text": text, "segments": seg_dicts}

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
