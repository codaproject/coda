__all__ = ["AudioProcessor", "Transcriber"]

import os
import logging
import tempfile

import gilda
import numpy as np
from scipy.io import wavfile

logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(self, sample_rate=16000, chunk_duration=3):
        """Initialize audio processor.

        Parameters
        ----------
        sample_rate :
            Audio sample rate (16000 Hz for Whisper)
        chunk_duration :
            Duration of audio chunks to process (seconds)
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = sample_rate * chunk_duration
        self.audio_buffer = np.array([], dtype=np.int16)

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

    def get_chunk(self) -> np.ndarray:
        """Get a chunk of audio for processing."""
        if len(self.audio_buffer) >= self.chunk_size:
            chunk = self.audio_buffer[:self.chunk_size]
            # Keep some overlap for better continuity (0.5 seconds)
            overlap_size = int(self.sample_rate * 0.5)
            self.audio_buffer = self.audio_buffer[self.chunk_size - overlap_size:]
            return chunk
        return None

    def clear_buffer(self):
        """Clear the audio buffer"""
        self.audio_buffer = np.array([], dtype=np.int16)


class Transcriber:
    async def transcribe_audio(self, audio_data: np.ndarray,
                               sample_rate: int = 16000):
        try:
            # Convert int16 to float32
            audio_float = audio_data.astype(np.float32) / 32768.0

            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                wavfile.write(tmp_file.name, sample_rate, audio_float)
                tmp_filename = tmp_file.name

            # Transcribe with Whisper
            result = self.transcribe_file(
                tmp_filename,
                language="en",  # Set to None for auto-detection
                fp16=False,
                verbose=False
            )

            # Clean up temp file
            os.unlink(tmp_filename)

            text = result["text"].strip()
            annotations = gilda.annotate(text)

            return text, annotations

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            # Show full trace
            import traceback
            traceback.print_exc()
            return "", {}

    async def transcribe_file(self, file_path: str, language: str = "en",
                              fp16: bool = False, verbose: bool = False):
        raise NotImplementedError


