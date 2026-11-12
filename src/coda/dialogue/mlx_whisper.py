"""Changes to how MLX Whisper decodes a buffer of audio.

MLX Whisper is the engine WhisperLiveKit runs on Apple Silicon. It decodes a
buffer, checks the result, and decodes the buffer again at a higher
temperature when the text is too repetitive or the model too unsure. The
function here changes how often it retries.
"""

# Temperatures a buffer is decoded at: the first one, then the next ones in
# turn while a decode fails the check
TEMPERATURES = (0.0, 0.2, 0.4)


def limit_retries(asr):
    """Retry a failed decode of an MLX LocalAgreement ASR twice, not five times.

    Each retry can take as long as a decode that runs to the token limit, and
    the transcript waits for all of them. The decode at the last temperature
    is returned even if it fails the check too.
    """
    decode = asr.model

    def wrapper(audio, **kwargs):
        return decode(audio, temperature=TEMPERATURES, **kwargs)

    asr.model = wrapper
