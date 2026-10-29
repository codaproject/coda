"""Benchmark ASR engines on the Zulu and Sesotho COD clips.

Whisper itself ships no model for either language, so the engines are MMS
language adapters, Whisper checkpoints fine-tuned on the African Next Voices
corpus and other community data, a w2v-BERT CTC fine-tune, and Omnilingual ASR,
with stock Whisper as an untuned baseline. Run through run_benchmark with
--language zu or st.
"""
import os

from engines import make_faster_whisper, make_hf_ctc, make_hf_whisper, \
    make_mms, make_omniasr

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

MMS_REPO = "facebook/mms-1b-all"
# MMS has no Sesotho adapter, so Sesotho is decoded with Northern Sotho
MMS_SUBSTITUTES = {"sot": "nso"}

# Single-language fine-tunes, keyed by ISO 639-3
ANV_REPOS = {
    "zul": "dsfsi-anv/whisper-large-v3-turbo-anv-zul",
    "sot": "dsfsi-anv/whisper-large-v3-turbo-anv-sot",
}
ANV_MULTILINGUAL = "dsfsi-anv/za-anv-multilingual-whisper-v3-turbo"
# The multilingual checkpoint ships weights without any tokenizer files
ANV_BASE = "openai/whisper-large-v3-turbo"
# A CTranslate2 conversion of the multilingual checkpoint
SWIVURISO = "digiphyte/swivuriso-turbo"

# Other community fine-tunes, keyed by ISO 639-3, as name to (repo, base). Some
# ship without tokenizer files, so the processor comes from the base model.
WHISPER_FINETUNES = {
    "zul": {
        "anv-zul-small": ("dsfsi-anv/whisper-small-anv-zulu-first-batch",
                          "openai/whisper-small"),
        "sitwala-anv-zul-250h-turbo": (
            "sitwala/whisper-large-v3-turbo-anv-zul-250h",
            "openai/whisper-large-v3-turbo"),
        "sitwala-nchlt-zul-turbo": ("sitwala/whisper-large-v3-turbo-nchlt-zul",
                                    "openai/whisper-large-v3-turbo"),
        "theirstory-zulu-medium": ("TheirStory/whisper-medium-zulu",
                                   "openai/whisper-medium"),
        "zionia-isizulu-small": ("zionia/whisper-small-isizulu",
                                 "openai/whisper-small"),
    },
    "sot": {
        "anv-sot-small": ("dsfsi-anv/whisper-small-anv-sot",
                          "openai/whisper-small"),
        "sitwala-anv-sot-large-v3": ("sitwala/whisper-large-v3-anv-sot",
                                     "openai/whisper-large-v3"),
        "sitwala-anv-sot-150h-turbo": (
            "sitwala/whisper-large-v3-turbo-anv-sot-150h",
            "openai/whisper-large-v3-turbo"),
        "sitwala-nchlt-sot-turbo": ("sitwala/whisper-large-v3-turbo-nchlt-sot",
                                    "openai/whisper-large-v3-turbo"),
        "misterkissi-sesotho-small": ("misterkissi/whisper-small-sesotho",
                                      "openai/whisper-small"),
        "toadoum-sesotho-small": ("Toadoum/whisper-small-sesotho",
                                  "openai/whisper-small"),
    },
}
CTC_FINETUNES = {
    "zul": {"w2v-bert-zulu": "badrex/w2v-bert-2.0-zulu-asr"},
    "sot": {},
}

# Stock Whisper has neither language, so these show the untuned baseline
STOCK_WHISPER = ("small", "large-v3-turbo", "large-v3")

OMNI_CARDS = {
    "omniasr-llm-300m": "omniASR_LLM_300M",
    "omniasr-llm-3b": "omniASR_LLM_3B",
}


def build_engines(args):
    language = args.language
    if language in MMS_SUBSTITUTES:
        mms_language = MMS_SUBSTITUTES[language]
        mms_name = f"mms-1b-all-{mms_language}"
    else:
        mms_language, mms_name = language, "mms-1b-all"
    engines = {mms_name: lambda: make_mms(MMS_REPO, mms_language, args.device)}
    # Whisper has no token for either language, so these decode on auto-detection
    engines[f"anv-{language}-turbo"] = \
        lambda: make_hf_whisper(ANV_REPOS[language], args.device)
    engines["anv-multilingual-turbo"] = \
        lambda: make_hf_whisper(ANV_MULTILINGUAL, args.device,
                                processor_repo=ANV_BASE)
    engines["swivuriso-turbo"] = \
        lambda: make_faster_whisper(SWIVURISO, None, args.fw_device,
                                    args.compute_type,
                                    condition_on_previous_text=False)
    for name, (repo, base) in WHISPER_FINETUNES[language].items():
        engines[name] = lambda repo=repo, base=base: \
            make_hf_whisper(repo, args.device, processor_repo=base)
    for name, repo in CTC_FINETUNES[language].items():
        engines[name] = lambda repo=repo: make_hf_ctc(repo, args.device)
    for size in STOCK_WHISPER:
        engines[f"whisper-{size}"] = lambda size=size: \
            make_faster_whisper(size, None, args.fw_device, args.compute_type,
                                condition_on_previous_text=False)
    # Omnilingual names a language by ISO 639-3 plus script
    for name, card in OMNI_CARDS.items():
        engines[name] = lambda card=card: \
            make_omniasr(card, f"{language}_Latn", args.device)
    return engines
