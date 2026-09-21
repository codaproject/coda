"""Benchmark ASR engines on the Tsonga COD clips.

Whisper itself ships no Tsonga model, so the engines here are MMS language
adapters, Whisper checkpoints fine-tuned on the African Next Voices corpus, and
Omnilingual ASR. Scoring, reporting and dataset loading are shared, see
run_benchmark and languages.ts. Run with no args for all engines, or pass engine
names. The Omnilingual engines pin old dependencies and need their own
environment.
"""
import os
import tempfile
import wave

import numpy as np

from engines import MIN_SAMPLES, SAMPLE_RATE, load_audio, \
    make_faster_whisper, make_hf_whisper
from run_benchmark import main_for

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# The fl102 checkpoint covers only the FLEURS languages and omits Tsonga
MMS_REPOS = {
    "mms-1b-all": "facebook/mms-1b-all",
    "mms-1b-l1107": "facebook/mms-1b-l1107",
}
# CTC over a whole clip at once is wasteful, so decode a window at a time
MMS_WINDOW_SEC = 30

ANV_TSO = "dsfsi-anv/whisper-large-v3-turbo-anv-tso"
ANV_MULTILINGUAL = "dsfsi-anv/za-anv-multilingual-whisper-v3-turbo"
# The multilingual checkpoint ships weights without any tokenizer files
ANV_BASE = "openai/whisper-large-v3-turbo"
# A CTranslate2 conversion of the multilingual checkpoint
SWIVURISO = "digiphyte/swivuriso-turbo"

# Only the LLM cards read a language code, the CTC ones discard it
OMNI_CARDS = {
    "omniasr-llm-300m": "omniASR_LLM_300M",
    "omniasr-llm-3b": "omniASR_LLM_3B",
}
# Omnilingual names a language by ISO 639-3 plus script
OMNI_LANG = "tso_Latn"
# The pipeline rejects anything longer than 40 seconds
OMNI_CHUNK_SEC = 35


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


def build_engines(args):
    engines = {
        name: lambda repo=repo: make_mms(repo, args.language, args.device)
        for name, repo in MMS_REPOS.items()
    }
    # Whisper has no Tsonga token, so these three decode on auto-detection
    engines["anv-tso-turbo"] = \
        lambda: make_hf_whisper(ANV_TSO, args.device)
    engines["anv-multilingual-turbo"] = \
        lambda: make_hf_whisper(ANV_MULTILINGUAL, args.device,
                                processor_repo=ANV_BASE)
    engines["swivuriso-turbo"] = \
        lambda: make_faster_whisper(SWIVURISO, None, args.fw_device,
                                    args.compute_type,
                                    condition_on_previous_text=False)
    for name, card in OMNI_CARDS.items():
        engines[name] = \
            lambda card=card: make_omniasr(card, OMNI_LANG, args.device)
    return engines


def main():
    return main_for("ts", build_engines)


if __name__ == "__main__":
    main()
