import asyncio
import contextlib
import json
from typing import AsyncIterator
from urllib.parse import quote

from websockets.asyncio.client import connect

from coda.config import settings
from . import StreamingTranscriber, TranscriptEvent
from .whisper_livekit import DEFAULT_MODEL_SIZE, _events_from_response


class WhisperLiveKitRemoteTranscriber(StreamingTranscriber):
    """Streaming backend that proxies to a native whisperlivekit-server.

    Forwards raw PCM (s16le, 16 kHz) to the server's /asr WebSocket and turns its
    responses into the same committed and preview TranscriptEvents as the
    in-process backend. This lets transcription run on host hardware (e.g.
    MLX/Metal) while the app runs in a container. The model is chosen when the
    server is launched, so the model argument here is ignored.
    """
    MODELS = ("tiny", "base", "small", "medium",
              "large", "large-v2", "large-v3")
    DEFAULT_MODEL = DEFAULT_MODEL_SIZE

    @classmethod
    def create(cls, model=None):
        return cls()

    def __init__(self, url: str = None):
        self.url = (url or settings.dialogue.transcriber_url).rstrip("/")

    async def stream(self, audio: AsyncIterator[bytes], *,
                     language: str = "en",
                     task: str = "transcribe") -> AsyncIterator[TranscriptEvent]:
        async with connect(f"{self.url}/asr?language={quote(language)}",
                           max_size=None) as ws:
            forwarder = asyncio.create_task(self._forward(ws, audio))
            state = {"emitted": {}, "preview": ""}
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "ready_to_stop":
                        break
                    if msg.get("type"):
                        continue
                    for event in _events_from_response(msg, state):
                        yield event
            finally:
                forwarder.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await forwarder

    async def _forward(self, ws, audio: AsyncIterator[bytes]):
        """Feed incoming audio to the server, then signal end-of-audio."""
        async for data in audio:
            await ws.send(data)
        with contextlib.suppress(Exception):
            await ws.send(b"")
