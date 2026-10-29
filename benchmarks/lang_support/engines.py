"""Engine factories shared by language-support benchmarks."""
import os
import subprocess
import tempfile
import wave

import numpy as np

SAMPLE_RATE = 16000
# Whisper encodes a fixed 30 second window
WHISPER_WINDOW_SEC = 30
# Chunks below this length carry no speech worth decoding
MIN_SAMPLES = SAMPLE_RATE // 10
# CTC over a whole clip at once is wasteful, so decode a window at a time
MMS_WINDOW_SEC = 30
# The Omnilingual pipeline rejects anything longer than 40 seconds
OMNI_CHUNK_SEC = 35


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


def write_wav(path, audio):
    """Write mono float32 samples as 16 bit PCM."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def make_omniasr(card, language, device=None, chunk_sec=OMNI_CHUNK_SEC):
    """Build a run(path) -> str closure over an Omnilingual ASR checkpoint.

    The pipeline takes files and refuses audio past its length limit, so a clip
    is cut into windows and transcribed as one batch.
    """
    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    # The default bfloat16 has no accelerated path outside CUDA
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipeline = ASRInferencePipeline(model_card=card, device=device, dtype=dtype)

    def transcribe(path):
        audio = load_audio(path)
        window = chunk_sec * SAMPLE_RATE
        with tempfile.TemporaryDirectory() as directory:
            chunks = []
            for start in range(0, len(audio), window):
                chunk = audio[start:start + window]
                if len(chunk) < MIN_SAMPLES:
                    continue
                name = os.path.join(directory, f"{start:09d}.wav")
                write_wav(name, chunk)
                chunks.append(name)
            texts = pipeline.transcribe(
                chunks, lang=[language] * len(chunks), batch_size=1)
        return " ".join(texts)

    return transcribe


def make_mms(repo, language, device):
    """Build a run(path) -> str closure over one MMS language adapter.

    The checkpoint holds a separate CTC head per language, so the head shipped
    in the base weights is replaced and its size deliberately mismatches.
    """
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    processor = AutoProcessor.from_pretrained(repo, target_lang=language)
    model = Wav2Vec2ForCTC.from_pretrained(
        repo, target_lang=language, ignore_mismatched_sizes=True
    ).to(device).eval()

    def transcribe(path):
        audio = load_audio(path)
        window = MMS_WINDOW_SEC * SAMPLE_RATE
        texts = []
        for start in range(0, len(audio), window):
            chunk = audio[start:start + window]
            if len(chunk) < MIN_SAMPLES:
                continue
            inputs = processor(chunk, sampling_rate=SAMPLE_RATE,
                               return_tensors="pt")
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits
            predicted = torch.argmax(logits, dim=-1)
            texts.append(processor.batch_decode(predicted)[0])
        return " ".join(texts)

    return transcribe


def make_hf_ctc(repo, device):
    """Build a run(path) -> str closure over a CTC checkpoint on the Hub."""
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(repo)
    model = AutoModelForCTC.from_pretrained(repo).to(device).eval()

    def transcribe(path):
        audio = load_audio(path)
        window = MMS_WINDOW_SEC * SAMPLE_RATE
        texts = []
        for start in range(0, len(audio), window):
            chunk = audio[start:start + window]
            if len(chunk) < MIN_SAMPLES:
                continue
            inputs = processor(chunk, sampling_rate=SAMPLE_RATE,
                               return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            predicted = torch.argmax(logits, dim=-1)
            texts.append(processor.batch_decode(predicted)[0])
        return " ".join(texts)

    return transcribe
