"""Benchmark ASR engines on the Tsonga COD clips.

Whisper itself ships no Tsonga model, so the engines here are MMS language
adapters, Whisper checkpoints fine-tuned on the African Next Voices corpus, and
Omnilingual ASR. Scoring, reporting and dataset loading are shared, see
run_benchmark and languages.ts. Run with no args for all engines, or pass engine
names. The Omnilingual engines pin old dependencies and need their own
environment.
"""
import os

from engines import make_faster_whisper, make_hf_whisper, make_mms, \
    make_omniasr
from run_benchmark import main_for

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# The fl102 checkpoint covers only the FLEURS languages and omits Tsonga
MMS_REPOS = {
    "mms-1b-all": "facebook/mms-1b-all",
    "mms-1b-l1107": "facebook/mms-1b-l1107",
}

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
