"""Per-case storage of interview data.

Each case gets its own folder under ``storage.output_dir`` holding a
``manifest.json`` plus one file per enabled ``storage.store`` item. A case is
recorded in one or more segments, one per connection: continuing a case
appends to its files and records the new segment's audio in its own WAV. An
inference made from clinical notes outside a connection is appended to its
case without adding a segment. A ``CaseRecorder`` is only created when
``storage.enabled`` is true.
``list_cases`` and ``load_case`` read the folders back for viewing.
"""

import json
import os
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


def storage_keeps_audio() -> bool:
    return storage_enabled() and \
        bool(settings.storage.store.get("audio", False))


def storage_keeps_notes() -> bool:
    return storage_enabled() and \
        bool(settings.storage.store.get("notes", False))


def cases_root() -> Path:
    output_dir = settings.storage.get("output_dir", "")
    if output_dir:
        return Path(output_dir).expanduser()
    return CODA_BASE.join(name="cases")


class CaseRecorder:
    """Writes everything recorded for one case into its own folder.

    With ``case_id`` the existing case is reopened and a new segment appended
    to it. The time between the two segments is recorded as a pause. Without
    ``audio`` the segment keeps no audio, as for a clinical-notes inference.
    A new case starts with ``metadata``, replaced by that given on close.
    """

    def __init__(
        self,
        session_id: str,
        generation: int,
        run_info: dict,
        case_id: str | None = None,
        audio: bool = True,
        metadata: dict | None = None,
    ):
        self.store = {key: bool(value)
                      for key, value in settings.storage.store.items()}
        keeps_audio = audio and self.store.get("audio", False)
        started = datetime.now()
        self.reopened = case_id is not None
        self._heard = False
        if case_id is None:
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
                "segments": [],
            }
            if self.store.get("metadata") and metadata:
                self.manifest["metadata"] = metadata
        else:
            self.case_id = case_id
            self.case_dir = cases_root() / case_id
            self.manifest = self._reopened_manifest(started)

        self.segment = len(self.manifest["segments"])
        audio_file = "audio.wav" if self.segment == 0 \
            else f"audio_{self.segment}.wav"
        self.manifest["segments"].append(
            {
                "session_id": session_id,
                "started_at": started.isoformat(),
                "ended_at": None,
                "coda_version": coda.__version__,
                "run_info": run_info,
                "audio": audio_file if keeps_audio else None,
            }
        )
        self._write_manifest()

        self._transcripts: dict[str, TextIO] = {}
        self._chunks = self._open("chunks.jsonl", "annotations")
        self._edits = self._open("edits.jsonl", "transcripts")
        self._inference = self._open("inference.jsonl", "inference")
        self._timing = self._open("timing.jsonl", "timing")
        self._audio: Optional[wave.Wave_write] = None
        if keeps_audio:
            self._audio = wave.open(str(self.case_dir / audio_file), "wb")
            self._audio.setnchannels(1)
            self._audio.setsampwidth(2)
            self._audio.setframerate(AUDIO_SAMPLE_RATE)

    def _reopened_manifest(self, resumed: datetime) -> dict:
        """The stored manifest, closed if a crash left it open, with the gap
        until ``resumed`` added as a pause."""
        last_write = _last_write(self.case_dir).isoformat()
        manifest = _read_manifest(self.case_dir)
        manifest["segments"] = _segments(manifest, self.case_dir)
        ended = manifest.get("ended_at") or last_write
        if manifest["segments"][-1]["ended_at"] is None:
            manifest["segments"][-1]["ended_at"] = ended
        for pause in manifest.setdefault("pauses", []):
            if pause["resumed_at"] is None:
                pause["resumed_at"] = ended
        self._gap_index = len(manifest["pauses"])
        manifest["pauses"].append(
            {"paused_at": ended, "resumed_at": resumed.isoformat(), "gap": True}
        )
        manifest["ended_at"] = None
        return manifest

    def _open(self, filename: str, store_key: str) -> TextIO | None:
        if not self.store.get(store_key):
            return None
        path = self.case_dir / filename
        if self.reopened:
            _end_last_line(path)
        return open(path, "a", encoding="utf-8")

    def _write_manifest(self):
        path = self.case_dir / "manifest.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, default=str)
        os.replace(tmp, path)

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
        self._heard = True
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

    def write_chunk(
        self,
        chunk_id: str,
        timestamp: float,
        english_text: str,
        annotations: list,
        timings: dict,
        original_text: str | None = None,
        original_language: str | None = None,
    ):
        received_at = time.time()
        if self.store.get("transcripts"):
            self._append_transcript(english_text, "en")
            if original_text and original_language:
                self._append_transcript(original_text, original_language)

        record: dict[str, Any] = {
            "chunk_id": chunk_id,
            "segment": self.segment,
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

    def write_edit(self, edit_id, through, before, after, annotations, live,
                   user=None):
        """Append an edit of the transcript inference reads to the case's
        edit log.

        The edit replaces the transcript up to and including chunk
        ``through``, ``before`` with ``after``. ``live`` is whether it was
        made while recording. The inference it leads to is written with the
        same ``edit_id``.
        """
        record = {
            "edit_id": edit_id,
            "through": through,
            "edited_at": time.time(),
            "live": live,
            "before": before,
            "after": after,
            "annotations": [a.to_json() for a in annotations],
        }
        if self.store.get("user"):
            record["user"] = user
        self._append_json(self._edits, record)

    def write_imported(self, text: str):
        """Write a transcript recorded elsewhere as one chunk of this case."""
        if self.store.get("transcripts"):
            self._append_transcript(text, "en")
        now = time.time()
        self._append_json(
            self._chunks,
            {
                "chunk_id": "imported",
                "segment": self.segment,
                "timestamp": now,
                "received_at": now,
                "text": text,
                "annotations": [],
                "imported": True,
            },
        )

    def write_inference(self, request: dict, result: dict, shown_at: float,
                        edit_id=None):
        """Append an inference result. One made from clinical notes keeps the
        notes' filenames, and their text if ``store.notes`` is set. One made
        after a transcript edit keeps its ``edit_id``."""
        notes = request.get("clinical_notes")
        record = {
            "shown_at": shown_at,
            "chunk_id": result.get("chunk_id", request.get("chunk_id")),
            "timestamp": result.get("timestamp", request.get("timestamp")),
            "chunks_processed": result.get("chunks_processed"),
            "source": "recording" if notes is None else "clinical_note",
            "causes": result.get("causes"),
            "reasoning": result.get("reasoning"),
            "questions": result.get("questions"),
        }
        if edit_id is not None:
            record["edit_id"] = edit_id
        if notes is not None:
            record["notes"] = [
                (
                    {"filename": n["filename"], "text": n["text"]}
                    if self.store.get("notes")
                    else {"filename": n["filename"]}
                )
                for n in notes
            ]
        if self.store.get("metadata"):
            record["metadata"] = request.get("metadata")
        self._append_json(self._inference, record)
        self._append_json(self._timing, {
            "kind": "inference", "chunk_id": record["chunk_id"],
            "at": shown_at, **(result.get("timings") or {}),
        })

    def close(self, metadata: dict | None = None, user: dict | None = None):
        for f in [*self._transcripts.values(), self._chunks, self._edits,
                  self._inference, self._timing]:
            if f is not None:
                f.close()
        self._transcripts.clear()
        if self._audio is not None:
            self._audio.close()
            self._audio = None
        if self.reopened and not self._heard:
            self._drop_segment()
            return
        ended = datetime.now().isoformat()
        self.manifest["ended_at"] = ended
        self.manifest["segments"][-1]["ended_at"] = ended
        if self.store.get("metadata"):
            self.manifest["metadata"] = metadata
        if self.store.get("user"):
            self.manifest["user"] = user
        self._write_manifest()

    def _drop_segment(self):
        """Leave the case as it was, as this segment received no audio."""
        segment = self.manifest["segments"].pop()
        if segment["audio"]:
            (self.case_dir / segment["audio"]).unlink(missing_ok=True)
        del self.manifest["pauses"][self._gap_index :]
        self.manifest["ended_at"] = self.manifest["segments"][-1]["ended_at"]
        self._write_manifest()


def _read_jsonl(path: Path) -> list[dict] | None:
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


def _end_last_line(path: Path):
    """End a line cut short by a crash, so the next record starts afresh."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        cut_short = f.read(1) != b"\n"
    if cut_short:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n")


def _last_write(case_dir: Path) -> datetime:
    return datetime.fromtimestamp(
        max(p.stat().st_mtime for p in case_dir.iterdir()))


def find_case(case_id: str) -> Path | None:
    """The folder of a stored case, or None if there is no such case."""
    if not CASE_ID_PATTERN.fullmatch(case_id):
        return None
    case_dir = cases_root() / case_id
    return case_dir if (case_dir / "manifest.json").is_file() else None


def _read_manifest(case_dir: Path) -> dict:
    return json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))


def _segments(manifest: dict, case_dir: Path) -> list[dict]:
    """The segments a case was recorded in. Cases recorded before segments
    were kept have one, described by the manifest itself."""
    if "segments" in manifest:
        return manifest["segments"]
    return [
        {
            "session_id": manifest.get("session_id"),
            "started_at": manifest["started_at"],
            "ended_at": manifest.get("ended_at"),
            "coda_version": manifest.get("coda_version"),
            "run_info": manifest.get("run_info"),
            "audio": "audio.wav" if (case_dir / "audio.wav").exists() else None,
        }
    ]


def _apply_edits(chunks, edits):
    """The chunks as shown after the last edit: one chunk holding its text,
    then the chunks after the last one it covered."""
    if not edits:
        return chunks
    edit = edits[-1]
    chunks = chunks or []
    ids = [chunk["chunk_id"] for chunk in chunks]
    covered = ids.index(edit["through"]) + 1 if edit["through"] in ids \
        else len(chunks)
    edited = {
        "chunk_id": edit["through"] or edit["edit_id"],
        "timestamp": edit["edited_at"],
        "text": edit["after"],
        "annotations": edit["annotations"],
        "segment": chunks[covered - 1].get("segment", 0) if covered else 0,
    }
    return [edited, *chunks[covered:]]


def _read_case(case_dir: Path) -> \
        tuple[dict, list[dict] | None, list[dict] | None]:
    return (
        _read_manifest(case_dir),
        _read_jsonl(case_dir / "chunks.jsonl"),
        _read_jsonl(case_dir / "inference.jsonl"),
    )


def _audio_seconds(path: Path) -> float | None:
    """Length of a WAV file, or None if it is missing or holds no frames."""
    try:
        with wave.open(str(path)) as wav:
            frames = wav.getnframes()
    except (FileNotFoundError, EOFError, wave.Error):
        return None
    return round(frames / AUDIO_SAMPLE_RATE, 1) if frames else None


def _parts(case_dir: Path, manifest: dict) -> list[dict]:
    """Each segment's times, settings and length of audio, if any."""
    return [
        {
            "part": i,
            "started_at": _with_utc_offset(segment["started_at"]),
            "ended_at": _with_utc_offset(segment["ended_at"])
            if segment["ended_at"] else None,
            "run_info": segment.get("run_info"),
            "audio_s": (
                _audio_seconds(case_dir / segment["audio"])
                if segment.get("audio")
                else None
            ),
        }
        for i, segment in enumerate(_segments(manifest, case_dir))
    ]


def _recording_seconds(manifest: dict, ended: datetime) -> float:
    """Time from start to ``ended``, less the time spent paused."""
    paused = sum(
        (
            (datetime.fromisoformat(p["resumed_at"])
             if p["resumed_at"] else ended)
            - datetime.fromisoformat(p["paused_at"])
        ).total_seconds()
        for p in manifest.get("pauses", [])
    )
    started = datetime.fromisoformat(manifest["started_at"])
    return max(0.0, (ended - started).total_seconds() - paused)


def _with_utc_offset(timestamp: str) -> str:
    """A manifest time, written in server-local time, with its UTC offset.

    Without the offset a browser in another time zone would read it as its own
    local time.
    """
    return datetime.fromisoformat(timestamp).astimezone().isoformat()


def _sources(chunks: list[dict], inference: list[dict]) -> list[str]:
    """What a case's inferences were made from: ``recording`` or
    ``clinical_note``. Empty for a case that has neither yet."""
    sources = {record.get("source", "recording") for record in inference}
    # Imported chunks are the transcript of an earlier recording too
    if chunks:
        sources.add("recording")
    return sorted(sources)


def case_sources(case_id: str) -> list[str]:
    """What a stored case's inferences were made from, empty if there is no
    such case."""
    case_dir = find_case(case_id)
    if case_dir is None:
        return []
    return _sources(
        _read_jsonl(case_dir / "chunks.jsonl") or [],
        _read_jsonl(case_dir / "inference.jsonl") or [],
    )


def _summary(
    case_dir: Path, manifest: dict, chunks: list[dict], inference: list[dict]
) -> dict:
    ended_at = manifest.get("ended_at")
    # A case that never closed ends at its last write
    ended = datetime.fromisoformat(ended_at) if ended_at \
        else _last_write(case_dir)
    causes = inference[-1].get("causes") if inference else None
    metadata = manifest.get("metadata")
    if metadata is None and inference:
        metadata = inference[-1].get("metadata")
    return {
        "case_id": case_dir.name,
        "started_at": _with_utc_offset(manifest["started_at"]),
        "ended_at": _with_utc_offset(ended_at) if ended_at else None,
        "duration_s": round(_recording_seconds(manifest, ended)),
        "language": (manifest.get("run_info") or {}).get("language"),
        "chunk_count": len(chunks),
        "inference_count": len(inference),
        "top_cause": (
            max(causes.values(), key=lambda c: c["score"]) if causes else None
        ),
        "has_audio": any(part["audio_s"]
                         for part in _parts(case_dir, manifest)),
        "sources": _sources(chunks, inference),
        "metadata": metadata,
        "user": manifest.get("user"),
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
        "annotations": [
            _display_annotation(ann)
            for ann in chunk.get("annotations", [])
            if ann.get("matches")
        ],
        "original_transcript": chunk.get("original_text"),
        "original_language": chunk.get("original_language"),
        "segment": chunk.get("segment", 0),
        "imported": chunk.get("imported", False),
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
        cases.append(_summary(case_dir, manifest, chunks or [],
                              inference or []))
    return sorted(cases, key=lambda c: datetime.fromisoformat(c["started_at"]),
                  reverse=True)


def load_case(case_id: str) -> dict | None:
    """Everything stored for one case, or None if there is no such case.

    ``chunks`` are as shown after the last transcript edit. ``chunks``,
    ``inference`` and ``edits`` are None when not stored. The
    case profile is written when a case closes, so for one that never closed
    it is taken from the last inference request. ``notes`` are the clinical
    notes of the last inference made from them.
    """
    case_dir = find_case(case_id)
    if case_dir is None:
        return None
    manifest, chunks, inference = _read_case(case_dir)
    edits = _read_jsonl(case_dir / "edits.jsonl")
    shown = _apply_edits(chunks, edits)
    note_inferences = [record for record in inference or []
                       if record.get("source") == "clinical_note"]
    return {
        **_summary(case_dir, manifest, chunks or [], inference or []),
        "coda_version": manifest.get("coda_version"),
        "run_info": manifest.get("run_info"),
        "pauses": manifest.get("pauses", []),
        "segments": _parts(case_dir, manifest),
        "chunks": [_display_chunk(c) for c in shown]
        if shown is not None else None,
        "transcripts": {
            path.stem.removeprefix("transcript_"):
            path.read_text(encoding="utf-8")
            for path in sorted(case_dir.glob("transcript_*.txt"))
        },
        "inference": inference,
        "edits": edits,
        "notes": note_inferences[-1].get("notes", [])
        if note_inferences else [],
    }


def case_audio_path(case_id: str, part: int = 0) -> Path | None:
    """The WAV recorded in one segment of a case, if it holds any audio."""
    case_dir = find_case(case_id)
    if case_dir is None:
        return None
    segments = _segments(_read_manifest(case_dir), case_dir)
    if not 0 <= part < len(segments) or not segments[part].get("audio"):
        return None
    path = case_dir / segments[part]["audio"]
    return path if _audio_seconds(path) else None
