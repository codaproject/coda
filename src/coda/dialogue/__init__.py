__all__ = ["AudioProcessor", "Transcriber", "ChunkedTranscriber",
           "TranscriptEvent", "create_transcriber", "TRANSCRIBER_BACKENDS"]

import os
import logging
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Tuple

import gilda
import numpy as np
from scipy.io import wavfile

logger = logging.getLogger(__name__)

# Default chunk length (seconds) used by the live app's audio pipeline.
DEFAULT_CHUNK_DURATION = 3

# Selectable transcription backends, in preference order (whisper is the
# default; faster-whisper and speechmatics are alternatives). The active
# backend is set by the TRANSCRIBER_BACKEND env var (see coda.runtime_config).
TRANSCRIBER_BACKENDS = ("whisper", "faster-whisper", "speechmatics")


def create_transcriber(backend: str = None, whisper_model: str = None):
    """Build a Transcriber for the named backend.

    Imports are deferred so a Speechmatics deployment needn't install
    torch/whisper, and vice versa. Defaults to the TRANSCRIBER_BACKEND env var.
    """
    from coda.runtime_config import get_transcriber_backend
    backend = (backend or get_transcriber_backend()).lower()
    if backend == "whisper":
        from .whisper import WhisperTranscriber, DEFAULT_MODEL_SIZE
        return WhisperTranscriber(
            model_size=whisper_model or DEFAULT_MODEL_SIZE
        )
    if backend == "faster-whisper":
        from .faster_whisper import FasterWhisperTranscriber, DEFAULT_MODEL_SIZE
        return FasterWhisperTranscriber(
            model_size=whisper_model or DEFAULT_MODEL_SIZE
        )
    if backend == "speechmatics":
        from .speechmatics import SpeechmaticsTranscriber
        return SpeechmaticsTranscriber()
    raise ValueError(
        f"Unknown transcriber backend {backend!r}; "
        f"choose from {TRANSCRIBER_BACKENDS}"
    )


class AudioProcessor:
    def __init__(self, sample_rate=16000, chunk_duration=DEFAULT_CHUNK_DURATION,
                 start_time=None):
        """Initialize audio processor.

        Parameters
        ----------
        sample_rate :
            Audio sample rate (16000 Hz for Whisper)
        chunk_duration :
            Duration of audio chunks to process (seconds)
        start_time :
            Unix time of the first captured sample; chunk timestamps are derived
            from this plus the chunk's position. Defaults to now.
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio_buffer = np.array([], dtype=np.int16)
        self.start_time = start_time if start_time is not None else time.time()
        self.samples_emitted = 0

    def add_audio(self, audio_data: bytes) -> bool:
        """Add audio data to buffer

        Returns
        -------
        True if buffer has enough data for processing
        """
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_array])

        # Check if we have enough audio for processing
        return len(self.audio_buffer) >= self.chunk_size

    def get_chunk(self) -> Optional[Tuple[str, float, np.ndarray]]:
        """Get a chunk of audio for processing with unique ID and timestamp.

        Returns
        -------
        Optional[Tuple[str, float, np.ndarray]]
            Tuple of (chunk_id, timestamp, audio_data) if chunk is ready, None otherwise
            timestamp is Unix time (seconds since epoch)
        """
        if len(self.audio_buffer) >= self.chunk_size:
            chunk_id = str(uuid.uuid4())
            # Timestamp from audio position, not processing time
            timestamp = self.start_time + self.samples_emitted / self.sample_rate
            chunk = self.audio_buffer[:self.chunk_size]
            self.audio_buffer = self.audio_buffer[self.chunk_size:]
            self.samples_emitted += self.chunk_size
            return (chunk_id, timestamp, chunk)
        return None

    def clear_buffer(self):
        """Clear the audio buffer"""
        self.audio_buffer = np.array([], dtype=np.int16)


@dataclass
class TranscriptEvent:
    """One unit of transcription output.

    Chunked backends emit one `committed=True` event per audio chunk. Streaming
    backends additionally emit `committed=False` interim previews.
    """
    id: str
    timestamp: float
    text: str
    committed: bool = True
    speaker: Optional[int] = None


class Transcriber:
    """Abstract transcription backend.

    Turns a stream of raw int16 PCM byte blobs into an async iterator of
    TranscriptEvents.
    """
    async def stream(self, audio: AsyncIterator[bytes], *,
                     language: str = "en",
                     task: str = "transcribe") -> AsyncIterator[TranscriptEvent]:
        raise NotImplementedError


class ChunkedTranscriber(Transcriber):
    """Transcriber that decodes fixed-length chunks independently.

    Buffers incoming audio into DEFAULT_CHUNK_DURATION-second chunks via
    AudioProcessor and transcribes each one, emitting a single committed event
    per chunk.
    """
    async def stream(self, audio: AsyncIterator[bytes], *,
                     language: str = "en",
                     task: str = "transcribe") -> AsyncIterator[TranscriptEvent]:
        processor = AudioProcessor()
        async for data in audio:
            if not processor.add_audio(data):
                continue
            while True:
                result = processor.get_chunk()
                if result is None:
                    break
                chunk_id, timestamp, chunk = result
                text = await self.transcribe_audio(
                    chunk, language=language, task=task
                )
                yield TranscriptEvent(id=chunk_id, timestamp=timestamp,
                                      text=text, committed=True)

    async def transcribe_audio(self, audio_data: np.ndarray,
                               sample_rate: int = 16000,
                               language: str = "en",
                               task: str = "transcribe") -> str:
        try:
            # Convert int16 to float32
            audio_float = audio_data.astype(np.float32) / 32768.0

            # Log audio diagnostics
            rms = float(np.sqrt(np.mean(audio_float ** 2)))
            peak = float(np.max(np.abs(audio_float)))
            logger.info(f"Audio chunk: {len(audio_float)} samples, "
                        f"rms={rms:.4f}, peak={peak:.4f}, "
                        f"language={language}, task={task}")
            if peak < 0.001:
                logger.warning("Audio appears to be silent (peak < 0.001)")
                return ""

            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                wavfile.write(tmp_file.name, sample_rate, audio_float)
                tmp_filename = tmp_file.name

            # Transcribe with Whisper
            result = await self.transcribe_file(
                tmp_filename,
                language=language,
                task=task,
                fp16=getattr(self, "fp16", False),
                verbose=False
            )

            # Clean up temp file
            os.unlink(tmp_filename)

            # Log raw result for debugging
            raw_text = result.get("text", "").strip()
            segments = result.get("segments", [])
            if segments:
                probs = [f"{s.get('no_speech_prob', 0):.2f}" for s in segments]
                logger.info(f"Raw transcription ({language}/{task}): "
                            f"{raw_text!r}")
                logger.info(f"Segment no_speech_probs: {probs}")
            else:
                logger.info(f"Raw transcription ({language}/{task}): "
                            f"{raw_text!r} (no segments)")

            # Filter segments based on no_speech_prob to avoid hallucinations
            # during silence (e.g., "thank you for watching").
            # Only applied for English - Whisper's no_speech_prob is
            # unreliable for other languages.
            text = self._filter_segments(result, language=language)
            return text

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            # Show full trace
            import traceback
            traceback.print_exc()
            return ""

    async def transcribe_file(self, file_path: str, language: str = "en",
                              task: str = "transcribe",
                              fp16: bool = False, verbose: bool = False):
        raise NotImplementedError

    def _filter_segments(self, result: dict, language: str = "en") -> str:
        """Extract text from transcription result.

        Subclasses may override this to filter segments based on
        backend-specific criteria.

        Parameters
        ----------
        result :
            The result dictionary from the transcription backend

        Returns
        -------
        str
            Transcription text
        """
        return result.get("text", "").strip()


