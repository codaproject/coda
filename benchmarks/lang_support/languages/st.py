"""Sesotho dataset loading and scoring normalization."""
import re
import unicodedata
from dataset_io import match_recordings, read_references

# ISO 639-3, the code MMS names its adapters by
ASR_LANGUAGE = "sot"

# Supplied filenames are inconsistent, so each recording is mapped by name
CASE_IDS = {
    "Ito-1 lower respiratory, infant.m4a.mp4": "lri_1",
    "Ito-2 lower respiratory infections.m4a.mp4": "lri_2",
    "Ito-2 lower respiratory infections child.m4a.mp4": "lri_3",
    "Malaria-1 Sesotho b54 child.m4a.mp4": "malaria_1",
    "Malaria-2 Sesotho b54 child.m4a.mp4": "malaria_2",
    "Diarrhoea-1 diarrhoea l deceased.m4a.mp4": "diarrhea_1",
    "Diarrhoea-2 A09.m4a.mp4": "diarrhea_2",
}


def normalize(text, strip_accents=False):
    """Lowercase Sesotho and replace punctuation, keeping hyphens as in e-na.

    strip_accents is accepted for a uniform signature across languages and has
    no meaning for Sesotho, which is written without diacritics.
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
        directory / "references" / "cases_st.json", ("st_narrative",)
    )
    return match_recordings(directory / "audio", references, case_id)
