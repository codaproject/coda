"""Brazilian Portuguese dataset loading and scoring normalization."""
import re
import unicodedata
from dataset_io import match_recordings, read_references

# Whisper-style code this dataset is transcribed with
ASR_LANGUAGE = "pt"


def normalize(text, strip_accents=False):
    """Lowercase Portuguese and optionally ignore accents while scoring."""
    text = text.lower()
    if strip_accents:
        text = "".join(
            char for char in unicodedata.normalize("NFD", text)
            if unicodedata.category(char) != "Mn"
        )
    text = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )
    return re.sub(r"\s+", " ", text).strip()


def case_id(path):
    # The supplied Iri filenames use capital I in place of lowercase l.
    stem = path.stem.lower()
    prefix, separator, number = stem.rpartition("_")
    return f"{'lri' if prefix == 'iri' else prefix}{separator}{number}"


def load_samples(directory):
    # Several Portuguese records retain the Bengali field name from the template.
    references = read_references(
        directory / "references" / "cases_ptbr_filtered.json",
        ("ptbr_narrative", "bn_narrative"),
    )
    return match_recordings(directory / "audio", references, case_id)
