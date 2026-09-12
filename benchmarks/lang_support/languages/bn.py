"""Bengali recordings supplied with cod-case_id_ filename prefixes."""
from dataset_io import match_recordings, read_references


def load_samples(directory):
    references = read_references(
        directory / "references" / "cases_bn_filtered.json", ("bn_narrative",)
    )
    return match_recordings(
        directory / "audio", references,
        lambda path: path.stem.removeprefix("cod-case_id_"),
    )
