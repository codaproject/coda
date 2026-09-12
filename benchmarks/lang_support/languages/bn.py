"""Bengali dataset loading and scoring normalization."""
import re
import unicodedata
from dataset_io import match_recordings, read_references


BENGALI_DIGIT_MAP = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
BENGALI_CARDINAL_WORDS = {
    "শূন্য": "0", "এক": "1", "দুই": "2", "দু": "2", "দো": "2",
    "তিন": "3", "চার": "4", "চারি": "4", "পাঁচ": "5",
    "ছয়": "6", "সাত": "7", "আট": "8",
    "দশ": "10", "এগারো": "11", "বারো": "12", "তেরো": "13",
    "চৌদ্দ": "14", "চোদ্দ": "14", "পনেরো": "15", "পনর": "15",
    "ষোলো": "16", "সতেরো": "17", "সতর": "17", "আঠারো": "18", "আঠেরো": "18",
    "ঊনিশ": "19", "ঊন্নিশ": "19", "বিশ": "20", "কুড়ি": "20", "একুশ": "21",
    "ত্রিশ": "30", "তিরিশ": "30", "চল্লিশ": "40", "পঞ্চাশ": "50",
    "ষাট": "60", "ষাটি": "60", "ষাইট": "60", "সত্তর": "70", "আশি": "80",
    "নব্বই": "90", "নব্বুই": "90", "শত": "100", "একশ": "100",
}


def normalize(text):
    """Preserve Bengali marks while removing punctuation and folding numerals."""
    text = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )
    text = re.sub(r"\s+", " ", text).strip().translate(BENGALI_DIGIT_MAP)
    return " ".join(BENGALI_CARDINAL_WORDS.get(word, word) for word in text.split())


def load_samples(directory):
    references = read_references(
        directory / "references" / "cases_bn_filtered.json", ("bn_narrative",)
    )
    return match_recordings(
        directory / "audio", references,
        lambda path: path.stem.removeprefix("cod-case_id_"),
    )
