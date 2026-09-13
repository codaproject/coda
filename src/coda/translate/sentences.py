"""Gating for translation over chunked transcription.

Sequence-to-sequence translation models are trained on whole sentences and
degrade on fragments, but chunked transcribers commit text on a fixed audio
cadence that cuts sentences apart. Committed chunks are joined here until a
sentence is available, with a word cap and an idle flush so speech that never
carries a terminator still gets translated.
"""
import re

# Sentence terminators, including the Bengali danda. A trailing ellipsis is
# excluded: transcribers emit it for hesitation mid-sentence, and treating it
# as a boundary chops the sentence before the speaker has finished it.
SENTENCE_END = re.compile(r"(?<!\.\.)[.!?।]\s*$")

# Translate a partial sentence once it reaches this many words, so a speaker
# who runs on without punctuation is not held indefinitely.
TRANSLATION_MAX_WORDS = 40

# Translate pending text that never reached a terminator after this many
# seconds idle, so a short trailing utterance is still translated.
TRANSLATION_MAX_WAIT_S = 6.0


class SentenceBuffer:
    """Accumulates committed chunk text until a full sentence is available.

    Callers add committed chunks, check `ready`, and `take()` the batch to
    translate (which resets the buffer). One batch can span several chunks, so
    the chunk ids it covers are returned with it.
    """

    def __init__(self, max_words=TRANSLATION_MAX_WORDS):
        self.max_words = max_words
        self._reset()

    def _reset(self):
        self.parts = []
        self.chunk_ids = []
        self.timestamp = None

    def add(self, text, chunk_id, timestamp):
        """Add one committed chunk; empty text is ignored."""
        if not text:
            return
        self.parts.append(text)
        self.chunk_ids.append(chunk_id)
        if self.timestamp is None:
            self.timestamp = timestamp

    @property
    def text(self):
        return " ".join(self.parts)

    @property
    def has_pending(self):
        return bool(self.parts)

    @property
    def ready(self):
        """Whether a sentence has closed, or the word cap has been reached."""
        if not self.parts:
            return False
        if SENTENCE_END.search(self.parts[-1].strip()):
            return True
        return len(self.text.split()) >= self.max_words

    def take(self):
        """Return (text, chunk_ids, timestamp) for the pending batch and reset,
        or None if nothing is pending."""
        if not self.parts:
            return None
        batch = (self.text, list(self.chunk_ids), self.timestamp)
        self._reset()
        return batch
