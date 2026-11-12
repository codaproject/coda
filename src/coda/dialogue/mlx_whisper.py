"""Changes to how MLX Whisper decodes a buffer of audio.

MLX Whisper is the engine WhisperLiveKit runs on Apple Silicon. It decodes a
buffer, checks the result, and decodes the buffer again at a higher
temperature when the text is too repetitive or the model too unsure. The
functions here change how often it retries and which decode it returns.
"""
import logging
import threading

logger = logging.getLogger(__name__)

# Temperatures a buffer is decoded at: the first one, then the next ones in
# turn while a decode fails the check
TEMPERATURES = (0.0, 0.2, 0.4)
# MLX Whisper's check of a decode, which CODA leaves at its defaults: the
# text is too repetitive above the first, the model unsure below the second
MAX_COMPRESSION_RATIO = 2.4
MIN_AVG_LOGPROB = -1.0
# Decodes of the buffer a thread is transcribing
DECODES = threading.local()


def limit_retries(asr):
    """Retry a failed decode of an MLX LocalAgreement ASR twice, not five times.

    Each retry can take as long as a decode that runs to the token limit, and
    the transcript waits for all of them.
    """
    decode = asr.model

    def wrapper(audio, **kwargs):
        return decode(audio, temperature=TEMPERATURES, **kwargs)

    asr.model = wrapper


def best_decode(results):
    """Return the most probable decode among those that are not repetitive.

    When all are repetitive, return the most probable of them, which is the
    choice faster-whisper makes when every decode of a buffer fails.
    """
    varied = [result for result in results
              if result.compression_ratio <= MAX_COMPRESSION_RATIO]
    return max(varied or results, key=lambda result: result.avg_logprob)


def keep_best_decode():
    """Have MLX Whisper return its best decode of a buffer when all fail.

    It returns the last one instead, the one at the highest temperature.
    """
    from mlx_whisper.whisper import Whisper
    decode = Whisper.decode
    if getattr(decode, "keeps_best", False):
        return

    def wrapper(self, mel, options, **kwargs):
        result = decode(self, mel, options, **kwargs)
        if isinstance(result, list):
            return result
        if options.temperature == TEMPERATURES[0]:
            DECODES.results = []
        results = getattr(DECODES, "results", [])
        results.append(result)
        failed = result.compression_ratio > MAX_COMPRESSION_RATIO \
            or result.avg_logprob < MIN_AVG_LOGPROB
        if not failed or options.temperature != TEMPERATURES[-1]:
            return result
        best = best_decode(results)
        logger.debug("Kept the decode at temperature %.1f of %d that failed",
                     best.temperature, len(results))
        return best

    wrapper.keeps_best = True
    Whisper.decode = wrapper
