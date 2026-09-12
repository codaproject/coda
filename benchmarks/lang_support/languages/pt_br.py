"""Brazilian Portuguese inputs, preserving the supplier's filenames and JSON."""
from dataset_io import match_recordings, read_references


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
