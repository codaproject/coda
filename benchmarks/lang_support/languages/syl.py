"""Sylheti dataset loading and scoring normalization."""
import re
from dataset_io import match_recordings, read_references
from languages import bn

# No engine has a Sylheti code, and it is written in Bengali script
ASR_LANGUAGE = "bn"

# Sylheti has no standard spelling. Letters for sounds that Sylheti does not
# distinguish are folded to one form: aspirated and plain stops, the
# affricates and sibilants, dental and retroflex nasals, flaps and r, and
# long and short vowels.
FOLD = str.maketrans({
    "খ": "ক", "ঘ": "গ", "ঝ": "জ", "ঠ": "ট", "ঢ": "ড",
    "থ": "ত", "ধ": "দ", "ফ": "প", "ভ": "ব",
    "ছ": "স", "চ": "স", "শ": "স", "ষ": "স",
    "ণ": "ন", "ঈ": "ই", "ঊ": "উ", "ী": "ি", "ূ": "ু",
})


def fold(word):
    # NFC keeps the flaps as a consonant plus nukta
    word = word.replace("ড়", "র").replace("ঢ়", "র")
    # Initial h is often dropped in speech, and a final o is written optionally
    word = re.sub(r"^হ", "অ", word)
    return re.sub(r"ো$", "", word.translate(FOLD))


SYLHETI_CARDINAL_WORDS = {"চাইর": "4"}


def normalize(text, strip_accents=False):
    return " ".join(fold(SYLHETI_CARDINAL_WORDS.get(word, word))
                    for word in bn.normalize(text).split())


def load_samples(directory):
    # Recordings are transcribed without a reference until one is supplied
    path = directory / "references" / "cases_syl.json"
    references = read_references(path, ("syl_narrative",)) \
        if path.exists() else None
    return match_recordings(directory / "audio", references,
                            lambda path: path.stem.removesuffix("_sn"))
