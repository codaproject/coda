"""Tsonga dataset loading and scoring normalization."""
import re
import unicodedata
from dataset_io import match_recordings, read_references

# ISO 639-3, the code MMS names its adapters by
ASR_LANGUAGE = "tso"

# Hyphen is word-internal in reduplications like hinkwako-nkwako
WORD_INTERNAL = "-"

# Dropped rather than kept, so n'wana and nwana score as the same one word
APOSTROPHES = "'’"

# Supplied recordings are numbered in the order of the references
CASE_IDS = {
    1: "lri_1",
    2: "lri_2",
    3: "lri_3",
    4: "malaria_1",
    5: "malaria_2",
    6: "diarrhea_1",
    7: "diarrhea_2",
}


def normalize(text, strip_accents=False):
    """Lowercase Tsonga, drop apostrophes, and keep word-internal hyphens.

    strip_accents is accepted for a uniform signature across languages and has
    no meaning for Tsonga, which is written without diacritics.
    """
    text = text.lower()
    for mark in APOSTROPHES:
        text = text.replace(mark, "")
    text = "".join(
        char if char in WORD_INTERNAL
        or not unicodedata.category(char).startswith("P") else " "
        for char in text
    )
    return re.sub(r"\s+", " ", text).strip()


def case_id(path):
    match = re.fullmatch(r"narrative\s*(\d+)", path.stem.lower())
    if not match:
        raise ValueError(f"Unrecognized recording name {path.name}")
    index = int(match.group(1))
    if index not in CASE_IDS:
        raise ValueError(f"Recording {path.name} is outside the reference set")
    return CASE_IDS[index]


def load_samples(directory):
    references = read_references(
        directory / "references" / "cases_ts.json", ("ts_narrative",)
    )
    return match_recordings(directory / "audio", references, case_id)
