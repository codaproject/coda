"""Translation through an NLLB sequence-to-sequence model under CTranslate2."""
import asyncio
import logging
import re
from functools import lru_cache

from . import Translator

logger = logging.getLogger(__name__)

# Sentence terminators, including the Bengali danda. NLLB is trained on single
# sentences and silently drops the tail of longer input, so text is split before
# translating and rejoined afterwards.
SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


@lru_cache(maxsize=1)
def _load(repo, device, compute_type):
    """Load a CTranslate2 model with its tokenizer, caching the last one used.

    Only one model is held at a time, so switching models reloads rather than
    keeping both resident.
    """
    import ctranslate2
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = snapshot_download(repo)
    translator = ctranslate2.Translator(path, device=device,
                                        compute_type=compute_type)
    return translator, AutoTokenizer.from_pretrained(path)


class CTranslate2Translator(Translator):
    """Translates a transcript to English with NLLB-200 under CTranslate2.

    The NLLB weights are released under CC-BY-NC, which rules out commercial
    deployment.
    """

    # Selectable models, surfaced in the settings UI
    MODELS = ("nllb-600m", "nllb-1.3b")
    DEFAULT_MODEL = "nllb-600m"
    REPOS = {
        "nllb-600m": "entai2965/nllb-200-distilled-600M-ctranslate2",
        "nllb-1.3b": "entai2965/nllb-200-distilled-1.3B-ctranslate2",
    }
    TARGET_LANGUAGE = "eng_Latn"
    # Whisper-style code to the FLORES-200 code NLLB expects. Where NLLB
    # ships a specific variant of a macrolanguage the common one is used,
    # for example Modern Standard Arabic and Simplified Chinese. Whisper
    # languages NLLB has no equivalent for are absent and report as
    # unsupported rather than being passed through silently.
    LANGUAGES = {
        "af": "afr_Latn", "am": "amh_Ethi", "ar": "arb_Arab", "as": "asm_Beng",
        "az": "azj_Latn", "ba": "bak_Cyrl", "be": "bel_Cyrl", "bg": "bul_Cyrl",
        "bn": "ben_Beng", "bo": "bod_Tibt", "bs": "bos_Latn", "ca": "cat_Latn",
        "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn", "de": "deu_Latn",
        "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn", "et": "est_Latn",
        "eu": "eus_Latn", "fa": "pes_Arab", "fi": "fin_Latn", "fo": "fao_Latn",
        "fr": "fra_Latn", "gl": "glg_Latn", "gu": "guj_Gujr", "ha": "hau_Latn",
        "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn", "ht": "hat_Latn",
        "hu": "hun_Latn", "hy": "hye_Armn", "id": "ind_Latn", "is": "isl_Latn",
        "it": "ita_Latn", "ja": "jpn_Jpan", "ka": "kat_Geor", "kk": "kaz_Cyrl",
        "km": "khm_Khmr", "kn": "kan_Knda", "ko": "kor_Hang", "lb": "ltz_Latn",
        "ln": "lin_Latn", "lo": "lao_Laoo", "lt": "lit_Latn", "lv": "lvs_Latn",
        "mi": "mri_Latn", "mk": "mkd_Cyrl", "ml": "mal_Mlym", "mn": "khk_Cyrl",
        "mr": "mar_Deva", "ms": "zsm_Latn", "mt": "mlt_Latn", "my": "mya_Mymr",
        "ne": "npi_Deva", "nl": "nld_Latn", "nn": "nno_Latn", "no": "nob_Latn",
        "oc": "oci_Latn", "pa": "pan_Guru", "pl": "pol_Latn", "pt": "por_Latn",
        "ro": "ron_Latn", "ru": "rus_Cyrl", "sa": "san_Deva", "sd": "snd_Arab",
        "si": "sin_Sinh", "sk": "slk_Latn", "sl": "slv_Latn", "sn": "sna_Latn",
        "so": "som_Latn", "sq": "als_Latn", "sr": "srp_Cyrl", "su": "sun_Latn",
        "sv": "swe_Latn", "sw": "swh_Latn", "ta": "tam_Taml", "te": "tel_Telu",
        "tg": "tgk_Cyrl", "th": "tha_Thai", "tk": "tuk_Latn", "tl": "tgl_Latn",
        "tr": "tur_Latn", "tt": "tat_Cyrl", "uk": "ukr_Cyrl", "ur": "urd_Arab",
        "uz": "uzn_Latn", "vi": "vie_Latn", "yi": "ydd_Hebr", "yo": "yor_Latn",
        "yue": "yue_Hant", "zh": "zho_Hans",
    }

    @classmethod
    def create(cls, model=None, device="cpu", compute_type="int8", **kwargs):
        return cls(model=model, device=device, compute_type=compute_type)

    def __init__(self, model=None, device="cpu", compute_type="int8"):
        self.model = model or self.DEFAULT_MODEL
        if self.model not in self.MODELS:
            raise ValueError(f"Unknown translation model {self.model!r}, "
                             f"expected one of {self.MODELS}")
        self.device = device
        self.compute_type = compute_type

    @classmethod
    def normalize_language(cls, code):
        """Resolve a transcription language code to a FLORES-200 code, or None."""
        if not code:
            return None
        return cls.LANGUAGES.get(code.strip())

    def _translate(self, text, source_code):
        translator, tokenizer = _load(self.REPOS[self.model], self.device,
                                      self.compute_type)
        tokenizer.src_lang = source_code
        sentences = [s for s in SENTENCE_END.split(text.strip()) if s.strip()]
        if not sentences:
            return ""
        tokens = [tokenizer.convert_ids_to_tokens(tokenizer.encode(s))
                  for s in sentences]
        results = translator.translate_batch(
            tokens, target_prefix=[[self.TARGET_LANGUAGE]] * len(tokens))
        return " ".join(
            tokenizer.decode(tokenizer.convert_tokens_to_ids(r.hypotheses[0][1:]))
            for r in results)

    async def translate(self, text, source_language, language_name=None):
        source_code = self.normalize_language(source_language)
        if source_code is None:
            logger.error("Language %r is not supported by NLLB, leaving the "
                         "text untranslated", source_language)
            return text
        try:
            # Decoding is synchronous and CPU-bound, keep it off the event loop
            return await asyncio.to_thread(self._translate, text, source_code)
        except Exception as e:
            logger.error(f"Translation error: {e}")
            # Fall back to the untranslated text
            return text
