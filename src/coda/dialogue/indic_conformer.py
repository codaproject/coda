"""Transcription backend for AI4Bharat's IndicConformer."""
import asyncio
import os
import logging

import numpy as np
import torch

from . import ChunkedTranscriber

logger = logging.getLogger(__name__)


class IndicConformerTranscriber(ChunkedTranscriber):
    """Transcriber using AI4Bharat's indic-conformer-600m-multilingual.

    A NeMo Conformer with hybrid CTC/RNNT decoding, see
    https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual

    The model consumes an audio tensor directly, so transcribe_audio() is
    overridden rather than going through the base class's temporary WAV file.
    """

    MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
    SAMPLE_RATE = 16000
    LANGUAGES = {
        "as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri",
        "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada", "kok": "Konkani",
        "ks": "Kashmiri", "mai": "Maithili", "ml": "Malayalam", "mni": "Manipuri",
        "mr": "Marathi", "ne": "Nepali", "or": "Odia", "pa": "Punjabi",
        "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "ta": "Tamil",
        "te": "Telugu", "ur": "Urdu",
    }

    # Decoding strategies
    MODELS = ("ctc", "rnnt")
    DEFAULT_MODEL = "ctc"

    @classmethod
    def create(cls, model=None):
        return cls(decoding=model or cls.DEFAULT_MODEL)

    def __init__(self, decoding=None):
        self.decoding = decoding or self.DEFAULT_MODEL
        if self.decoding not in self.MODELS:
            raise ValueError(f"Unknown decoding strategy {self.decoding!r}, "
                             f"expected one of {self.MODELS}")

        from transformers import AutoModel

        # None lets Hugging Face use credentials saved by `hf auth login`
        hf_token = os.environ.get("HF_TOKEN") or None
        self.model = AutoModel.from_pretrained(
            self.MODEL_ID, trust_remote_code=True, token=hf_token
        )

    def _decode(self, wav, language):
        with torch.no_grad():
            result = self.model(wav, language, self.decoding)
        if isinstance(result, (list, tuple)):
            result = result[0]
        return str(result).strip()

    async def transcribe_audio(self, audio_data, sample_rate=16000,
                               language="en", task="transcribe"):
        # The model has no translate mode, so task is ignored
        try:
            if sample_rate != self.SAMPLE_RATE:
                raise ValueError(f"Expected {self.SAMPLE_RATE} Hz audio, "
                                 f"got {sample_rate}")

            im_language = self.normalize_language(language)
            if im_language is None:
                logger.error(
                    "Language %r is not supported by IndicConformer, skipping chunk",
                    language,
                )
                return ""

            audio_float = audio_data.astype(np.float32) / 32768.0

            peak = float(np.max(np.abs(audio_float))) if audio_float.size else 0.0
            if peak < 0.001:
                logger.warning("Audio appears to be silent (peak < 0.001)")
                return ""

            wav = torch.from_numpy(audio_float).unsqueeze(0)
            # Inference is synchronous and CPU-bound, keep it off the event loop
            return await asyncio.to_thread(self._decode, wav, im_language)

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return ""
