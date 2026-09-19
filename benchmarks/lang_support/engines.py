"""Engine factories shared by language-support benchmarks."""
import subprocess

import numpy as np

SAMPLE_RATE = 16000
# Whisper encodes a fixed 30 second window
WHISPER_WINDOW_SEC = 30
# Chunks below this length carry no speech worth decoding
MIN_SAMPLES = SAMPLE_RATE // 10


def load_audio(path, sr=SAMPLE_RATE):
    """Decode any delivered format to mono float32 at sr via ffmpeg."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", str(path), "-f", "f32le", "-ac", "1",
         "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def make_hf_whisper(repo, device, processor_repo=None):
    """Build a run(path) -> str closure over a Whisper checkpoint on the Hub.

    Takes no language, since a fine-tune for a language outside Whisper's own
    set has no token to force. Pass processor_repo for a checkpoint whose
    weights ship without tokenizer files.
    """
    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    processor = AutoProcessor.from_pretrained(processor_repo or repo)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        repo, dtype=torch.float32).to(device).eval()
    model.generation_config.forced_decoder_ids = None

    def transcribe(path):
        audio = load_audio(path)
        window = WHISPER_WINDOW_SEC * SAMPLE_RATE
        texts = []
        for start in range(0, len(audio), window):
            chunk = audio[start:start + window]
            if len(chunk) < MIN_SAMPLES:
                continue
            features = processor(
                chunk, sampling_rate=SAMPLE_RATE,
                return_tensors="pt").input_features.to(device).to(model.dtype)
            with torch.no_grad():
                ids = model.generate(features, max_new_tokens=440)
            texts.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
        return " ".join(texts)

    return transcribe


def make_whisper(size, language, device):
    import whisper

    model = whisper.load_model(size, device=device)

    def transcribe(path):
        return model.transcribe(
            str(path), language=language, fp16=(device != "cpu")
        )["text"]

    return transcribe


def make_faster_whisper(size, language, device, compute_type,
                        condition_on_previous_text=True):
    """Build a run(path) -> str closure over a CTranslate2 Whisper build.

    Turn off condition_on_previous_text for a checkpoint whose decode loops.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(size, device=device, compute_type=compute_type)

    def transcribe(path):
        segments, _ = model.transcribe(
            str(path), language=language,
            condition_on_previous_text=condition_on_previous_text)
        return " ".join(segment.text for segment in segments)

    return transcribe
