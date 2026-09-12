"""
IndicConformerTranscriber

MODELS/create() follow Transcriber's contract (settings-UI
selectable options), matching SpeechmaticsTranscriber's MODELS pattern.
"""
import os
import logging

import numpy as np
import torch

from . import ChunkedTranscriber

logger = logging.getLogger(__name__)


class IndicConformerTranscriber(ChunkedTranscriber):
    """Transcriber using AI4Bharat's indic-conformer-600m-multilingual
    (NeMo Conformer, hybrid CTC/RNNT -- see
    https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual).

    Overrides transcribe_audio() entirely rather than using
    ChunkedTranscriber's file-path-based default, matching
    SpeechmaticsTranscriber's pattern -- this model takes a raw audio
    tensor directly, no temp file needed.
    """

    MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
    LANGUAGES = {
        "as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri",
        "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada", "kok": "Konkani",
        "ks": "Kashmiri", "mai": "Maithili", "ml": "Malayalam", "mni": "Manipuri",
        "mr": "Marathi", "ne": "Nepali", "or": "Odia", "pa": "Punjabi",
        "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "ta": "Tamil",
        "te": "Telugu", "ur": "Urdu",
    }

    # Selectable decoding strategies.
    MODELS = ("ctc", "rnnt")
    DEFAULT_MODEL = "ctc"  # project decision: CTC confirmed as default

    @classmethod
    def create(cls, model=None):
        return cls(decoding=model or cls.DEFAULT_MODEL)

    def __init__(self, decoding: str = None):

        from coda.config import settings
        _ = settings.dialogue.transcriber_backend   # force dynaconf's .env load
        
        self.decoding = decoding or self.DEFAULT_MODEL
        if self.decoding not in self.MODELS:
            raise ValueError(f"Unknown decoding strategy {self.decoding!r}; "
                              f"expected one of {self.MODELS}")

        from transformers import AutoModel

        # None lets Hugging Face use credentials saved by `hf auth login`.
        hf_token = os.environ.get("HF_TOKEN") or None
        self.model = AutoModel.from_pretrained(
            self.MODEL_ID, trust_remote_code=True, token=hf_token
        )

    async def transcribe_audio(self, audio_data: np.ndarray,
                               sample_rate: int = 16000,
                               language: str = "en",
                               task: str = "transcribe") -> str:
        # task accepted for signature compatibility with stream()'s calling
        # convention, ignored -- no translate mode for this model.
        try:
            im_language = self.normalize_language(language)
            if im_language is None:
                logger.error(
                    "Language %r is not supported by IndicConformer; skipping chunk",
                    language,
                )
                return ""

            # AudioProcessor buffers at 16kHz by construction (confirmed
            # via its own sample_rate=16000 default) -- no resampling needed.
            audio_float = audio_data.astype(np.float32) / 32768.0

            peak = float(np.max(np.abs(audio_float))) if audio_float.size else 0.0
            if peak < 0.001:  # matches ChunkedTranscriber's own default threshold
                logger.warning("Audio appears to be silent (peak < 0.001)")
                return ""

            wav = torch.from_numpy(audio_float).unsqueeze(0)
            with torch.no_grad():
                result = self.model(wav, im_language, self.decoding)
            if isinstance(result, (list, tuple)):
                result = result[0]
            return str(result).strip()

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return ""