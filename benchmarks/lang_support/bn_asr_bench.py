"""Benchmark Bangla ASR engines on the COD audio clips.

Whisper-based models run through the transformers ASR pipeline with long-form
chunking, indic-seamless and indic-conformer use their own code paths. Scoring,
reporting and dataset loading are shared, see run_benchmark and languages.bn.
Run with no args for all engines, or pass engine names.
"""
import os
import subprocess

import numpy as np

from coda.config import settings
from run_benchmark import main_for

# Initialize shared configuration before model libraries read environment variables.
settings.validators.validate()

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_audio(path, sr=16000):
    """Decode any format to mono float32 at sr via ffmpeg (avoids soundfile/m4a
    and torchcodec issues in the transformers pipeline)."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", str(path), "-f", "f32le", "-ac", "1",
         "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def make_whisper(repo, base_config_repo=None):
    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoConfig
    dev = _device()
    proc = AutoProcessor.from_pretrained(repo)
    if base_config_repo:
        config = AutoConfig.from_pretrained(base_config_repo)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            repo, config=config, dtype=torch.float32
        ).to(dev).eval()
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(repo, dtype=torch.float32).to(dev).eval()

    # These checkpoints ship an outdated generation config, so generate()
    # rejects the language argument and ignores forced_decoder_ids. They are
    # single-language fine-tunes, so decoding is left to auto-detection.
    model.generation_config.forced_decoder_ids = None

    def run(path):
        audio = load_audio(path)
        win = 30 * 16000
        texts = []
        for i in range(0, len(audio), win):
            chunk = audio[i:i + win]
            if len(chunk) < 1600:
                continue
            feats = proc(chunk, sampling_rate=16000,
                         return_tensors="pt").input_features.to(dev).to(model.dtype)
            with torch.no_grad():
                ids = model.generate(feats, max_new_tokens=440)
            texts.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
        return " ".join(texts)
    return run


def make_seamless(repo, tgt_lang="ben"):
    import os
    from transformers import AutoProcessor, SeamlessM4Tv2Model
    token = os.environ.get("HF_TOKEN")
    proc = AutoProcessor.from_pretrained(repo, token=token)
    model = SeamlessM4Tv2Model.from_pretrained(repo, token=token).to(_device())

    def run(path):
        inputs = proc(audio=load_audio(path), sampling_rate=16000,
                      return_tensors="pt").to(_device())
        out = model.generate(**inputs, tgt_lang=tgt_lang, generate_speech=False)
        return proc.decode(out[0].tolist()[0], skip_special_tokens=True)
    return run


def make_conformer(decoding="ctc"):
    """Build a run(path) -> str closure using the IndicConformerTranscriber
    class from src/coda/dialogue/indic_conformer.py
    """
    import asyncio
    from coda.dialogue.indic_conformer import IndicConformerTranscriber

    transcriber = IndicConformerTranscriber.create(model=decoding)

    def run(path):
        audio_float = load_audio(path)  # float32 mono @ 16kHz via ffmpeg

        # REQUIRED: transcribe_audio expects int16 PCM, not float32 [-1,1] casting directly truncates instead of scaling.
        pcm = (np.clip(audio_float, -1.0, 1.0) * 32767.0).astype(np.int16)

        return asyncio.run(
            transcriber.transcribe_audio(pcm, sample_rate=16000, language="bn", task="transcribe")
        )
    return run


def make_bst(size):
    from banglaspeech2text import Speech2Text
    stt = Speech2Text(size)
    return lambda path: stt.recognize(path)


def make_mlx(repo):
    import mlx_whisper
    return lambda path: mlx_whisper.transcribe(
        path, path_or_hf_repo=repo, language="bn")["text"]

def make_speechmatics(model="enhanced"):
    """Build a run(path) -> str closure using the SpeechmaticsTranscriber
    class from src/coda/dialogue/speechmatics.py.

    SpeechmaticsTranscriber.transcribe_audio() is async; this script's
    ENGINES closures are synchronous, so asyncio.run() bridges the two.
    """
    import asyncio
    from coda.dialogue.speechmatics import SpeechmaticsTranscriber

    transcriber = SpeechmaticsTranscriber.create(model=model)

    def run(path):
        audio_float = load_audio(path)
        # transcribe_audio expects int16 PCM, casting float32 directly would
        # truncate rather than scale
        pcm = (np.clip(audio_float, -1.0, 1.0) * 32767.0).astype(np.int16)

        return asyncio.run(
            transcriber.transcribe_audio(
                pcm, sample_rate=16000, language="bn", task="transcribe"
            )
        )
    return run


ENGINES = {
    "mlx-whisper-small": lambda: make_mlx("mlx-community/whisper-small-mlx"),
    "banglaspeech2text-base": lambda: make_bst("base"),
    "banglaspeech2text-large": lambda: make_bst("large"),
    # A Hindi fine-tune, kept as a cross-script control. It recognizes Bengali
    # speech but emits Devanagari, so nearly every word scores as a
    # substitution. Its own config.json does not match its weights, hence the
    # replacement config.
    "indic-whisper": lambda: make_whisper(
        "parthiv11/indic_whisper_nodcil",
        base_config_repo="openai/whisper-large-v2"),
    "tugstugi-regional-medium": lambda: make_whisper(
        "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"),
    "tugstugi-medium": lambda: make_whisper(
        "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"),
    "bangla-whisper-large-v3": lambda: make_whisper(
        "utshobs/bangla_whisper_large_v3_finetuned"),
    "indic-seamless": lambda: make_seamless("ai4bharat/indic-seamless"),
    "indic-conformer": lambda: make_conformer(decoding="ctc"),
    "speechmatics-enhanced": lambda: make_speechmatics("enhanced"),
    "speechmatics-standard": lambda: make_speechmatics("standard"),
}


def build_engines(args):
    return ENGINES


def main():
    return main_for("bn", build_engines)


if __name__ == "__main__":
    main()
