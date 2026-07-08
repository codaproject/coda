"""Gating for inference over streaming transcription.

Streaming transcribers commit many small segments; running an inference on each
one floods the agent and lets it fall behind the audio. Both the web app and the
offline CLI accumulate committed text and infer once enough new words arrive, or
after an idle timeout for a short trailing utterance. This module holds the
shared thresholds and the accumulation bookkeeping so the two paths stay in sync.
"""

# Infer once this many new words of committed transcript have accumulated. A
# chunked (non-streaming) backend emits whole chunks and infers per chunk, which
# is min_words=0 (always ready).
INFERENCE_MIN_WORDS = 15

# Flush pending text that never reached the word threshold after this many
# seconds idle, so a short final utterance still gets inferred.
INFERENCE_MAX_WAIT_S = 10.0


class StreamingInferenceBuffer:
    """Accumulates committed transcript text and annotations until enough new
    words have arrived to warrant an inference. Callers add committed segments,
    check `ready`, and `take()` the batch to infer on (which resets the buffer).
    """

    def __init__(self, min_words: int = INFERENCE_MIN_WORDS):
        self.min_words = min_words
        self._reset()

    def _reset(self):
        self.text = []
        self.anns = []
        self.words = 0
        self.chunk_id = None
        self.timestamp = None

    def add(self, text, anns, chunk_id, timestamp):
        """Add one committed segment; empty text is ignored."""
        if not text:
            return
        self.text.append(text)
        self.anns.extend(anns)
        self.words += len(text.split())
        self.chunk_id = chunk_id
        self.timestamp = timestamp

    @property
    def has_pending(self) -> bool:
        return bool(self.text)

    @property
    def ready(self) -> bool:
        """Whether enough new words have accumulated to infer."""
        return self.words >= self.min_words

    def take(self):
        """Return (text, anns, chunk_id, timestamp) for the pending batch and
        reset, or None if nothing is pending."""
        if not self.text:
            return None
        batch = (" ".join(self.text), list(self.anns),
                 self.chunk_id, self.timestamp)
        self._reset()
        return batch
