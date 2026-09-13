"""Shared representation and validation for language benchmark inputs."""
from dataclasses import dataclass
import json
import subprocess
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    case_id: str
    audio_path: Path
    reference: str


def read_references(path, text_keys):
    """Read supplied JSON without changing it; accept documented field aliases."""
    cases = {}
    for row in json.loads(path.read_text(encoding="utf-8")):
        case_id = row["case_id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Invalid case_id in {path}")
        if case_id in cases:
            raise ValueError(f"Duplicate case_id {case_id!r} in {path}")
        values = [row[k] for k in text_keys if row.get(k)]
        if not values or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f"Missing or invalid reference for {case_id!r} in {path}")
        if len(set(values)) != 1:
            raise ValueError(f"Conflicting reference fields for {case_id!r} in {path}")
        cases[case_id] = values[0]
    return cases


def match_recordings(audio_dir, references, identify):
    """Require one recording per supplied reference and reject unmatched audio."""
    samples = {}
    for path in sorted(audio_dir.glob("*.m4a")):
        case_id = identify(path)
        if case_id not in references:
            raise ValueError(f"No reference for recording {path}")
        if case_id in samples:
            raise ValueError(f"Duplicate recording for {case_id!r}: {path}")
        samples[case_id] = Sample(case_id, path, references[case_id])
    missing = references.keys() - samples.keys()
    if missing:
        raise ValueError(f"Missing recordings in {audio_dir}: {sorted(missing)}")
    if not samples:
        raise ValueError(f"No samples in {audio_dir}")
    return [samples[k] for k in sorted(samples)]


def load_samples(language, data_dir=None):
    """Load a dataset identified by its BCP 47 language tag."""
    from languages import bn, pt_br

    loaders = {"bn": bn.load_samples, "pt_br": pt_br.load_samples}
    if language not in loaders:
        raise ValueError(f"Unknown dataset language {language!r}; choose {list(loaders)}")
    data_dir = Path(data_dir) if data_dir is not None else Path(__file__).parent / "data"
    return loaders[language](data_dir / language)


def clip_duration(path):
    """Return a recording's duration in seconds, or None if ffprobe fails."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except Exception:
        return None
