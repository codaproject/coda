import json
import logging
import os

import numpy as np
import websockets

from coda.runtime_config import get_speechmatics_model, get_speechmatics_url

from . import ChunkedTranscriber
from .util import normalize_language

logger = logging.getLogger(__name__)


class SpeechmaticsTranscriber(ChunkedTranscriber):
    """Transcriber using Speechmatics' realtime WebSocket API.

    Each chunk is sent over a short-lived connection: StartRecognition,
    stream the raw PCM, EndOfStream, then collect the finalized transcript.
    """
    # Speechmatics selects an "operating point" rather than a Whisper size.
    MODELS = ("enhanced", "standard")
    DEFAULT_MODEL = get_speechmatics_model()

    @classmethod
    def create(cls, model=None):
        return cls(model=model or cls.DEFAULT_MODEL)

    def __init__(self, api_key: str = None, url: str = None,
                 model: str = None):
        self.api_key = api_key or self._resolve_api_key()
        if not self.api_key:
            raise ValueError(
                "No Speechmatics API key; set SPEECHMATICS_API_KEY or pass api_key"
            )
        self.url = url or get_speechmatics_url()
        self.model = model or get_speechmatics_model()

    @staticmethod
    def _resolve_api_key() -> str:
        """Read the key from SPEECHMATICS_API_KEY or a speechmatics_api_key file."""
        key = os.environ.get("SPEECHMATICS_API_KEY")
        if key:
            return key.strip()
        path = os.environ.get("SPEECHMATICS_API_KEY_FILE", "speechmatics_api_key")
        if os.path.exists(path):
            with open(path) as fh:
                return fh.read().strip()
        return ""

    async def transcribe_audio(self, audio_data: np.ndarray,
                               sample_rate: int = 16000,
                               language: str = "en",
                               task: str = "transcribe") -> str:
        try:
            sm_language = normalize_language(language)
            if sm_language is None:
                logger.error(
                    "Language %r is not supported by Speechmatics; skipping chunk",
                    language,
                )
                return ""
            pcm = np.ascontiguousarray(audio_data, dtype=np.int16)
            peak = int(np.max(np.abs(pcm))) if pcm.size else 0
            if peak < 33:
                logger.warning("Audio appears to be silent (peak < 0.001)")
                return ""
            return await self._stream(pcm.tobytes(), sample_rate, sm_language, task)
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return ""

    async def _stream(self, pcm_bytes: bytes, sample_rate: int,
                      language: str, task: str) -> str:
        """Run one StartRecognition -> EndOfStream -> EndOfTranscript exchange."""
        config = {
            "language": language,
            "operating_point": self.model,
        }
        # Speechmatics translates via translation_config, not a "translate" task.
        translate = task == "translate" and language != "en"
        if translate:
            config_message = {
                "message": "StartRecognition",
                "audio_format": {
                    "type": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": sample_rate,
                },
                "transcription_config": config,
                "translation_config": {"target_languages": ["en"]},
            }
        else:
            config_message = {
                "message": "StartRecognition",
                "audio_format": {
                    "type": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": sample_rate,
                },
                "transcription_config": config,
            }

        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with websockets.connect(
            self.url, additional_headers=headers
        ) as ws:
            await ws.send(json.dumps(config_message))

            await self._await_message(ws, "RecognitionStarted")

            # Stream audio as binary frames; count them for last_seq_no.
            seq = 0
            chunk_size = sample_rate * 2
            for start in range(0, len(pcm_bytes), chunk_size):
                await ws.send(pcm_bytes[start:start + chunk_size])
                seq += 1

            await ws.send(json.dumps(
                {"message": "EndOfStream", "last_seq_no": seq}
            ))

            transcripts = []
            final_key = "AddTranslation" if translate else "AddTranscript"
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                kind = msg["message"]
                if kind == final_key:
                    transcripts.append(self._text_from(msg, translate))
                elif kind == "Error":
                    logger.error(f"Speechmatics error: {msg}")
                    break
                elif kind == "EndOfTranscript":
                    break

            return " ".join(t for t in transcripts if t).strip()

    @staticmethod
    def _text_from(msg: dict, translate: bool) -> str:
        """Extract text from an AddTranscript or AddTranslation message."""
        if translate:
            # Translation results carry text in results[].content, no transcript field.
            return " ".join(
                r.get("content", "") for r in msg.get("results", [])
            ).strip()
        return msg.get("metadata", {}).get("transcript", "").strip()

    @staticmethod
    async def _await_message(ws, expected: str):
        """Read messages until the expected one arrives, raising on Error."""
        async for raw in ws:
            if isinstance(raw, bytes):
                continue
            msg = json.loads(raw)
            if msg["message"] == expected:
                return msg
            if msg["message"] == "Error":
                raise RuntimeError(f"Speechmatics error: {msg}")
