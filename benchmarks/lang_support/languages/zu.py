"""Zulu dataset loading and scoring normalization."""
import re
import unicodedata
from dataset_io import match_recordings, read_references

# ISO 639-3, the code MMS names its adapters by
ASR_LANGUAGE = "zul"

# Supplied filenames are inconsistent, so each recording is mapped by name
CASE_IDS = {
    "Iri-1 Zulu lower.m4a.mp4": "lri_1",
    "Ito-2 lower respiratory infections Zulu.m4a.mp4": "lri_2",
    "Lungs 3.m4a.mp4": "lri_3",
    "Malaria 1.m4a.mp4": "malaria_1",
    "Malaria rec 1.m4a.mp4": "malaria_2",
    "Diarrhoea-1 Zulu diarrhoea l diseases.m4a.mp4": "diarrhea_1",
    "Diarrhoea rec2.m4a.mp4": "diarrhea_2",
}

# Recordings of a different translation from the reference, left unscored
MISMATCHED = {"lri_1"}

# Recordings that open with a spoken title missing from the reference
SPOKEN_TITLE = {"lri_2", "diarrhea_1"}


def normalize(text, strip_accents=False):
    """Lowercase Zulu and replace punctuation, keeping word-internal hyphens.

    strip_accents is accepted for a uniform signature across languages and has
    no meaning for Zulu, which is written without diacritics.
    """
    text = "".join(
        char if char == "-" or not unicodedata.category(char).startswith("P")
        else " " for char in text.lower()
    )
    return re.sub(r"\s+", " ", text).strip()


def case_id(path):
    if path.name not in CASE_IDS:
        raise ValueError(f"Unrecognized recording name {path.name}")
    return CASE_IDS[path.name]


def load_samples(directory):
    references = read_references(
        directory / "references" / "cases_zu.json", ("zu_narrative",)
    )
    return match_recordings(directory / "audio", references, case_id)
