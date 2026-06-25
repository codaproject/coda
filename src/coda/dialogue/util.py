"""Speechmatics realtime language support.

Codes are the values accepted by transcription_config.language. The dropdown
(via the app's /languages endpoint) is built from SPEECHMATICS_LANGUAGES when
the Speechmatics backend is active, so users only see supported languages.
"""

# code -> display name for languages supported by Speechmatics realtime.
SPEECHMATICS_LANGUAGES = {
    "en": "English",
    "ar": "Arabic",
    "ba": "Bashkir",
    "eu": "Basque",
    "be": "Belarusian",
    "bn": "Bengali",
    "bg": "Bulgarian",
    "yue": "Cantonese",
    "ca": "Catalan",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "eo": "Esperanto",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "ia": "Interlingua",
    "ga": "Irish",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "ms": "Malay",
    "mt": "Maltese",
    "cmn": "Mandarin",
    "mr": "Marathi",
    "mn": "Mongolian",
    "no": "Norwegian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovakian",
    "sl": "Slovenian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "ug": "Uyghur",
    "vi": "Vietnamese",
    "cy": "Welsh",
}

# Forgive the few Whisper codes that differ from Speechmatics' codes, so a
# language carried over from the Whisper backend still resolves.
WHISPER_TO_SPEECHMATICS = {
    "zh": "cmn",
    "nn": "no",
    "nb": "no",
    "jw": "id",
}


def normalize_language(code):
    """Map a code to its Speechmatics equivalent, or None if unsupported."""
    if not code:
        return None
    code = code.strip()
    code = WHISPER_TO_SPEECHMATICS.get(code, code)
    return code if code in SPEECHMATICS_LANGUAGES else None
