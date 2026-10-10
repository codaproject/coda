"""Per-case storage of interview data.

Each case gets its own folder under ``storage.output_dir`` holding a
``manifest.json`` plus one file per enabled ``storage.store`` item. A
``CaseRecorder`` is only created when ``storage.enabled`` is true.
``list_cases`` and ``load_case`` read the folders back for read-only viewing.
"""

import json
import re
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TextIO

from gilda.term import get_curie

import coda
from coda import CODA_BASE
from coda.config import settings

AUDIO_SAMPLE_RATE = 16000
CASE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


def storage_enabled() -> bool:
    return bool(settings.storage.get("enabled", False))


def cases_browsable() -> bool:
    return bool(settings.storage.get("browse", False))


def cases_root() -> Path:
    output_dir = settings.storage.get("output_dir", "")
    if output_dir:
        return Path(output_dir).expanduser()
    return CODA_BASE.join(name="cases")


class CaseRecorder:
    """Writes everything recorded for one case into its own folder."""

    def __init__(self, session_id: str, generation: int, run_info: dict):
        self.store = {key: bool(value)
                      for key, value in settings.storage.store.items()}
        started = datetime.now()
        self.case_id = (f"{started.strftime('%Y-%m-%d_%H%M%S')}"
                        f"_{session_id[:8]}_g{generation}")
        self.case_dir = cases_root() / self.case_id
        self.case_dir.mkdir(parents=True, exist_ok=True)

        self.manifest = {
            "case_id": self.case_id,
            "session_id": session_id,
            "generation": generation,
            "started_at": started.isoformat(),
            "ended_at": None,
            "pauses": [],
            "coda_version": coda.__version__,
            "run_info": run_info,
        }
        self._write_manifest()

        self._transcripts: dict[str, TextIO] = {}
        self._chunks = self._open("chunks.jsonl", "annotations")
        self._inference = self._open("inference.jsonl", "inference")
        self._timing = self._open("timing.jsonl", "timing")
        self._audio: Optional[wave.Wave_write] = None
        if self.store.get("audio"):
            self._audio = wave.open(str(self.case_dir / "audio.wav"), "wb")
            self._audio.setnchannels(1)
            self._audio.setsampwidth(2)
            self._audio.setframerate(AUDIO_SAMPLE_RATE)

    def _open(self, filename: str, store_key: str) -> Optional[TextIO]:
        if not self.store.get(store_key):
            return None
        return open(self.case_dir / filename, "a", encoding="utf-8")

    def _write_manifest(self):
        with open(self.case_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    @staticmethod
    def _append_json(f: Optional[TextIO], record: dict):
        if f is None:
            return
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()

    def _append_transcript(self, text: str, lang_code: str):
        f = self._transcripts.get(lang_code)
        if f is None:
            f = open(self.case_dir / f"transcript_{lang_code}.txt", "a",
                     encoding="utf-8")
            self._transcripts[lang_code] = f
        f.write(text + "\n")
        f.flush()

    def write_audio(self, data: bytes):
        if self._audio is not None:
            self._audio.writeframes(data)

    def write_pause(self):
        self.manifest["pauses"].append(
            {"paused_at": datetime.now().isoformat(), "resumed_at": None})
        self._write_manifest()

    def write_resume(self):
        pauses = self.manifest["pauses"]
        if pauses and pauses[-1]["resumed_at"] is None:
            pauses[-1]["resumed_at"] = datetime.now().isoformat()
            self._write_manifest()

    def write_chunk(self, chunk_id: str, timestamp: float, english_text: str,
                    annotations: list, timings: dict,
                    original_text: Optional[str] = None,
                    original_language: Optional[str] = None):
        received_at = time.time()
        if self.store.get("transcripts"):
            self._append_transcript(english_text, "en")
            if original_text and original_language:
                self._append_transcript(original_text, original_language)

        record: dict[str, Any] = {
            "chunk_id": chunk_id,
            "timestamp": timestamp,
            "received_at": received_at,
            "text": english_text,
            "annotations": [a.to_json() for a in annotations] if annotations else [],
        }
        if original_text:
            record["original_text"] = original_text
            record["original_language"] = original_language
        self._append_json(self._chunks, record)
        self._append_json(self._timing, {
            "kind": "chunk", "chunk_id": chunk_id, "at": received_at, **timings,
        })

    def write_inference(self, request: dict, result: dict, shown_at: float):
        record = {
            "shown_at": shown_at,
            "chunk_id": result.get("chunk_id", request.get("chunk_id")),
            "timestamp": result.get("timestamp", request.get("timestamp")),
            "chunks_processed": result.get("chunks_processed"),
            "causes": result.get("causes"),
            "reasoning": result.get("reasoning"),
            "questions": result.get("questions"),
        }
        if self.store.get("metadata"):
            record["metadata"] = request.get("metadata")
        self._append_json(self._inference, record)
        self._append_json(self._timing, {
            "kind": "inference", "chunk_id": record["chunk_id"],
            "at": shown_at, **(result.get("timings") or {}),
        })

    def close(self, metadata: Optional[dict] = None):
        for f in [*self._transcripts.values(), self._chunks, self._inference,
                  self._timing]:
            if f is not None:
                f.close()
        self._transcripts.clear()
        if self._audio is not None:
            self._audio.close()
            self._audio = None
        self.manifest["ended_at"] = datetime.now().isoformat()
        if self.store.get("metadata"):
            self.manifest["metadata"] = metadata
        self._write_manifest()


def _read_jsonl(path: Path) -> Optional[list[dict]]:
    """Records in a JSONL file, or None if it was not stored.

    A line cut short by a server that stopped mid-write is skipped.
    """
    if not path.exists():
        return None
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    return records


def _case_dir(case_id: str) -> Optional[Path]:
    if not CASE_ID_PATTERN.fullmatch(case_id):
        return None
    case_dir = cases_root() / case_id
    return case_dir if (case_dir / "manifest.json").is_file() else None


def _read_case(case_dir: Path) -> tuple[dict, Optional[list[dict]],
                                        Optional[list[dict]]]:
    manifest = json.loads(
        (case_dir / "manifest.json").read_text(encoding="utf-8"))
    return (manifest, _read_jsonl(case_dir / "chunks.jsonl"),
            _read_jsonl(case_dir / "inference.jsonl"))


def _has_audio(case_dir: Path) -> bool:
    """Whether the case has an audio file holding any frames."""
    try:
        with wave.open(str(case_dir / "audio.wav")) as wav:
            return wav.getnframes() > 0
    except (FileNotFoundError, EOFError, wave.Error):
        return False


def _recording_seconds(manifest: dict, ended: datetime) -> float:
    """Time from start to ``ended``, less the time spent paused."""
    paused = sum(
        ((datetime.fromisoformat(p["resumed_at"]) if p["resumed_at"] else ended)
         - datetime.fromisoformat(p["paused_at"])).total_seconds()
        for p in manifest.get("pauses", [])
    )
    started = datetime.fromisoformat(manifest["started_at"])
    return max(0.0, (ended - started).total_seconds() - paused)


def _summary(case_dir: Path, manifest: dict, chunks: list[dict],
             inference: list[dict]) -> dict:
    ended_at = manifest.get("ended_at")
    # A case that never closed ends at its last write
    ended = (datetime.fromisoformat(ended_at) if ended_at else
             datetime.fromtimestamp(
                 max(p.stat().st_mtime for p in case_dir.iterdir())))
    causes = inference[-1].get("causes") if inference else None
    return {
        "case_id": case_dir.name,
        "started_at": manifest["started_at"],
        "ended_at": ended_at,
        "duration_s": round(_recording_seconds(manifest, ended)),
        "language": (manifest.get("run_info") or {}).get("language"),
        "chunk_count": len(chunks),
        "inference_count": len(inference),
        "top_cause": (max(causes.values(), key=lambda c: c["score"])
                      if causes else None),
        "has_audio": _has_audio(case_dir),
    }


def _display_annotation(annotation: dict) -> dict:
    """The fields the UI highlights, from a stored gilda annotation."""
    term = annotation["matches"][0]["term"]
    return {
        "text": annotation["text"],
        "start": annotation["start"],
        "end": annotation["end"],
        "curie": get_curie(term["db"], term["id"]),
        "name": term["entry_name"],
    }


def _display_chunk(chunk: dict) -> dict:
    """A stored chunk in the shape of the live ``transcript`` message."""
    return {
        "chunk_id": chunk["chunk_id"],
        "timestamp": chunk["timestamp"],
        "transcript": chunk["text"],
        "annotations": [_display_annotation(ann)
                        for ann in chunk.get("annotations", [])
                        if ann.get("matches")],
        "original_transcript": chunk.get("original_text"),
        "original_language": chunk.get("original_language"),
    }


def list_cases() -> list[dict]:
    """Summaries of the stored cases that recorded anything, newest first."""
    root = cases_root()
    if not root.is_dir():
        return []
    cases = []
    for manifest_path in root.glob("*/manifest.json"):
        case_dir = manifest_path.parent
        if not CASE_ID_PATTERN.fullmatch(case_dir.name):
            continue
        try:
            manifest, chunks, inference = _read_case(case_dir)
        except ValueError:
            continue
        if not (chunks or inference or any(case_dir.glob("transcript_*.txt"))):
            continue
        cases.append(_summary(case_dir, manifest, chunks or [], inference or []))
    return sorted(cases, key=lambda c: c["started_at"], reverse=True)


def load_case(case_id: str) -> Optional[dict]:
    """Everything stored for one case, or None if there is no such case.

    ``chunks`` and ``inference`` are None when that item was not stored. The
    case profile is written when a case closes, so for one that never closed
    it is taken from the last inference request.
    """
    case_dir = _case_dir(case_id)
    if case_dir is None:
        return None
    manifest, chunks, inference = _read_case(case_dir)
    metadata = manifest.get("metadata")
    if metadata is None and inference:
        metadata = inference[-1].get("metadata")
    return {
        **_summary(case_dir, manifest, chunks or [], inference or []),
        "coda_version": manifest.get("coda_version"),
        "run_info": manifest.get("run_info"),
        "pauses": manifest.get("pauses", []),
        "metadata": metadata,
        "chunks": ([_display_chunk(c) for c in chunks]
                   if chunks is not None else None),
        "transcripts": {
            path.stem.removeprefix("transcript_"):
                path.read_text(encoding="utf-8")
            for path in sorted(case_dir.glob("transcript_*.txt"))
        },
        "inference": inference,
    }


def case_audio_path(case_id: str) -> Optional[Path]:
    case_dir = _case_dir(case_id)
    if case_dir is None or not _has_audio(case_dir):
        return None
    return case_dir / "audio.wav"
