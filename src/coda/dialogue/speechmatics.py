import asyncio
import contextlib
import json
import logging
import os
import time
import uuid

import numpy as np
import websockets

from coda.config import settings

from . import StreamingTranscriber, TranscriptEvent

logger = logging.getLogger(__name__)


class SpeechmaticsTranscriber(StreamingTranscriber):
    """Transcriber using Speechmatics' realtime WebSocket API.

    A recording is streamed over one connection: StartRecognition, the raw
    PCM as it arrives, EndOfStream, then the remaining results.
    """
    LANGUAGES = {
        "en": "English",
        "ar": "Arabic",
        "ba": "Bashkir",
        "eu": "Basque",
        "be": "Belarusian",
        "bn": "Bengali",
        "bg": "Bulgarian",
        "yue": "Cantonese",
        "ca": "Catalan",
        "hr": "Croatian",
        "cs": "Czech",
        "da": "Danish",
        "nl": "Dutch",
        "eo": "Esperanto",
        "et": "Estonian",
        "fi": "Finnish",
        "fr": "French",
        "gl": "Galician",
        "de": "German",
        "el": "Greek",
        "he": "Hebrew",
        "hi": "Hindi",
        "hu": "Hungarian",
        "id": "Indonesian",
        "ia": "Interlingua",
        "ga": "Irish",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "lv": "Latvian",
        "lt": "Lithuanian",
        "ms": "Malay",
        "mt": "Maltese",
        "cmn": "Mandarin",
        "mr": "Marathi",
        "mn": "Mongolian",
        "no": "Norwegian",
        "fa": "Persian",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "sk": "Slovakian",
        "sl": "Slovenian",
        "es": "Spanish",
        "sw": "Swahili",
        "sv": "Swedish",
        "tl": "Tagalog",
        "ta": "Tamil",
        "th": "Thai",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "ug": "Uyghur",
        "vi": "Vietnamese",
        "cy": "Welsh",
    }

    LANGUAGE_ALIASES = {
        "zh": "cmn",
        "nn": "no",
        "nb": "no",
        "jw": "id",
    }

    # Speechmatics selects an "operating point" rather than a Whisper size.
    MODELS = ("enhanced", "standard")
    DEFAULT_MODEL = settings.dialogue.speechmatics.model

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
        self.url = url or settings.dialogue.speechmatics.url
        self.model = model or settings.dialogue.speechmatics.model

    @staticmethod
    def _resolve_api_key() -> str:
        """Resolve the key from SPEECHMATICS_API_KEY, a key file, or config secrets.

        SPEECHMATICS_API_KEY keeps its conventional (non-CODA-prefixed) name.
        """
        key = os.environ.get("SPEECHMATICS_API_KEY")
        if key:
            return key.strip()
        path = os.environ.get("SPEECHMATICS_API_KEY_FILE", "speechmatics_api_key")
        if os.path.exists(path):
            with open(path) as fh:
                return fh.read().strip()
        return (settings.dialogue.speechmatics.get("api_key") or "").strip()

    async def transcribe_audio(self, audio_data, sample_rate=16000,
                               language="en", task="transcribe"):
        """Transcribe one whole recording in a single session."""
        pcm = np.ascontiguousarray(audio_data, dtype=np.int16).tobytes()

        async def audio():
            yield pcm

        texts = []
        try:
            async for event in self.stream(audio(), language=language,
                                           task=task, sample_rate=sample_rate):
                if event.committed:
                    texts.append(event.text)
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
        return " ".join(texts).strip()

    async def stream(self, audio, *, language="en", task="transcribe",
                     sample_rate=16000):
        """Transcribe raw 16-bit PCM over one realtime session.

        Partial results are emitted as previews and final ones as committed
        events. Speechmatics translates via translation_config rather than a
        transcription task.
        """
        sm_language = self.normalize_language(language)
        if sm_language is None:
            logger.error("Language %r is not supported by Speechmatics",
                         language)
            async for _ in audio:
                pass
            return
        translate = task == "translate" and sm_language != "en"
        start = {
            "message": "StartRecognition",
            "audio_format": {"type": "raw", "encoding": "pcm_s16le",
                             "sample_rate": sample_rate},
            "transcription_config": {"language": sm_language,
                                     "operating_point": self.model,
                                     "enable_partials": True},
        }
        if translate:
            start["translation_config"] = {"target_languages": ["en"],
                                           "enable_partials": True}
        final, partial = ("AddTranslation", "AddPartialTranslation") \
            if translate else ("AddTranscript", "AddPartialTranscript")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with websockets.connect(self.url,
                                      additional_headers=headers) as ws:
            await ws.send(json.dumps(start))
            await self._await_message(ws, "RecognitionStarted")
            sender = asyncio.create_task(self._send_audio(ws, audio))
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    msg = json.loads(raw)
                    kind = msg["message"]
                    if kind in (final, partial):
                        text = self._text_from(msg, translate)
                        if text or kind == partial:
                            yield TranscriptEvent(
                                id=uuid.uuid4().hex, timestamp=time.time(),
                                text=text, committed=kind == final)
                    elif kind == "Error":
                        raise RuntimeError(f"Speechmatics error: {msg}")
                    elif kind == "EndOfTranscript":
                        break
            finally:
                sender.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await sender

    @staticmethod
    async def _send_audio(ws, audio):
        """Forward audio frames, then end the stream."""
        seq = 0
        async for data in audio:
            await ws.send(data)
            seq += 1
        await ws.send(json.dumps({"message": "EndOfStream",
                                  "last_seq_no": seq}))

    @staticmethod
    def _text_from(msg, translate):
        """Extract text from a transcript or translation message."""
        if translate:
            # Translation results carry text in results[].content
            return " ".join(
                r.get("content", "") for r in msg.get("results", [])
            ).strip()
        return msg.get("metadata", {}).get("transcript", "").strip()

    @staticmethod
    async def _await_message(ws, expected):
        """Read messages until the expected one arrives, raising on Error."""
        async for raw in ws:
            if isinstance(raw, bytes):
                continue
            msg = json.loads(raw)
            if msg["message"] == expected:
                return msg
            if msg["message"] == "Error":
                raise RuntimeError(f"Speechmatics error: {msg}")
