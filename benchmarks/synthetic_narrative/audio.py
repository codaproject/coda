"""16 kHz mono int16 WAV helpers, the format the live app sends to the transcriber."""
import contextlib
import wave
from pathlib import Path

import numpy as np

from common import SAMPLE_RATE


def read_wav(path):
    with contextlib.closing(wave.open(str(path), "rb")) as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != \
                (1, 2, SAMPLE_RATE):
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit WAV")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def write_wav(path, pcm):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(wave.open(str(path), "wb")) as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())


def seconds(pcm):
    return len(pcm) / SAMPLE_RATE
