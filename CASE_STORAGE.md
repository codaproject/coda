# CODA case storage

This document describes the files CODA writes for each case and the
application log. Case storage is off unless `storage.enabled` is true. Every
site profile turns it on.

## Folders

| What | Location |
|---|---|
| Case folders (Mac install) | `~/CODA_Cases/<case folder>/`, opened by the `CODA_Cases` shortcut on the Desktop |
| Case folders (other installs) | `~/.data/coda/cases/<case folder>/`, or `storage.output_dir` when set |
| Last issued case number | `last_case_number` in the folder holding the case folders |
| Application log (Mac install) | `~/Library/Logs/CODA/coda.log` |
| Application log (other installs) | stderr, and `logging.file` when set |
| Laptop settings (Mac install) | `~/Library/Application Support/CODA/settings.yaml` |

`~` is the home folder of the user running CODA. Updates never touch the
cases, the log or the laptop settings.

A case is one session in the app, from the first recording or clinical note
up to the next Reset. Continuing a stored case from History adds to the same
folder.

## Case folder name

```
<YYYY-MM-DD>_<HHMMSS>_<session>_g<generation>
```

For example `2026-11-10_203425_8ee901bd_g0`. The date and time are when the
case started, in the laptop's local time. `<session>` is the first 8
characters of the browser session ID, and `<generation>` counts the Resets
within that session, starting at 0.

The folder name is the storage ID of the case. The case ID entered in the app
(for example `CODA-007`) is stored in `manifest.json` under
`metadata.case_id`.

## Files in a case folder

| File | Format | Contents | Setting |
|---|---|---|---|
| `manifest.json` | JSON object | Case summary, settings and model versions, case profile, user | always written |
| `transcript_en.txt` | UTF-8 text | English transcript | `store.transcripts` |
| `transcript_<lang>.txt` | UTF-8 text | Transcript in the spoken language, for example `transcript_ts.txt` | `store.transcripts` |
| `edits.jsonl` | JSON Lines | Corrections made to the transcript | `store.transcripts` |
| `chunks.jsonl` | JSON Lines | Each piece of transcript with its grounding annotations | `store.annotations` |
| `inference.jsonl` | JSON Lines | Each inference result shown, with follow-up questions | `store.inference` |
| `timing.jsonl` | JSON Lines | Processing times per chunk and per inference | `store.timing` |
| `revisions.jsonl` | JSON Lines | One line per change to the case | always written |
| `audio.wav` | WAV | Audio of the first recording | `store.audio` |
| `audio_<n>.wav` | WAV | Audio of recording `<n>` of a continued case, from 1 | `store.audio` |

`store.metadata` controls whether the case profile is stored (in
`manifest.json`, `inference.jsonl` and `revisions.jsonl`), and `store.user`
controls whether the user's name and ID are stored (in `manifest.json`,
`edits.jsonl` and `revisions.jsonl`). `store.notes` controls whether the text
of clinical notes is kept in `inference.jsonl`. Their filenames are always
kept.

A file only exists once something was written to it. A case made from
clinical notes has no audio or transcript files.

## Formats

- **JSON Lines** (`.jsonl`): one JSON object per line, UTF-8, appended as
  the case goes on. If CODA stops in the middle of a write, the last line can
  be cut short. Readers should skip lines that don't parse.
- **Text** (`.txt`): UTF-8, one line per transcribed piece, in the order
  spoken.
- **WAV**: 16 kHz, mono, 16-bit PCM. Paused time is not recorded.
- **Times** come in two forms. `manifest.json` and `revisions.jsonl` use ISO
  8601 in the laptop's local time with no time zone, for example
  `2026-11-10T20:34:25.318152`. All other files use Unix time, seconds since
  1970-01-01 UTC, as a decimal number, for example `1793247136.74`.
- `null` marks a value that is unknown or wasn't entered.

## Schemas

### manifest.json

| Field | Type | Description |
|---|---|---|
| `case_id` | string | Case folder name |
| `session_id` | string | Browser session ID |
| `generation` | integer | Resets within the session before this case |
| `started_at` | ISO time | Start of the case |
| `ended_at` | ISO time or null | End of the last recording, null while recording |
| `pauses` | list of pause | Pauses, and gaps between recordings of a continued case |
| `coda_version` | string | CODA version at the start of the case |
| `coda_commit` | string | CODA commit at the start of the case |
| `run_info` | run info | Settings and models at the start of the case |
| `segments` | list of segment | One per recording or notes session |
| `metadata` | metadata | Case ID and profile, latest version |
| `user` | user | Person using the app when the case last changed |
| `revision` | integer | Number of the latest revision |

**Pause**

| Field | Type | Description |
|---|---|---|
| `paused_at` | ISO time | Start of the pause |
| `resumed_at` | ISO time or null | End of the pause, null if the recording stopped while paused |
| `gap` | boolean | Present and true for the time between two recordings of a continued case |

**Segment**

| Field | Type | Description |
|---|---|---|
| `session_id` | string | Browser session ID of this recording |
| `started_at` | ISO time | Start of the recording |
| `ended_at` | ISO time or null | End of the recording |
| `coda_version` | string | CODA version used |
| `coda_commit` | string | CODA commit used |
| `run_info` | run info | Settings and models used |
| `audio` | string or null | Audio filename, null when no audio was kept |

**Run info**

| Field | Type | Description |
|---|---|---|
| `site` | string | Site, for example `agincourt` |
| `usecase` | string | `clinical` or `interview` |
| `language` | string | Spoken language code, for example `ts` |
| `translation_mode` | string | How speech is turned into English |
| `transcriber_backend` | string | Speech recognition engine |
| `transcriber_model` | string | Speech recognition model |
| `grounder` | string | `gilda` or `rag` |
| `rag` | object or null | RAG grounder settings, when used |
| `translation_llm` | object | `provider` and `model` of the LLM used for translation, when that mode is used |
| `inference` | object or null | Inference agent: `agent`, `agent_classes`, `default_cohort`, `provider`, `model`, `num_questions`, `question_style`, `question_selection`. Null if the agent couldn't be reached |

**Metadata**

| Field | Type | Description |
|---|---|---|
| `case_id` | string or null | Case ID entered in the app, for example `CODA-007` |
| `profile` | profile or null | Decedent profile |

**Profile**

| Field | Type | Values |
|---|---|---|
| `cohort` | string | `child`, `adult` |
| `age` | object or null | `value` (number) and `unit` (`hours`, `days`, `weeks`, `months`, `years`) |
| `stillbirth` | boolean | |
| `date_of_death` | string or null | `YYYY-MM-DD` |
| `sex` | string or null | `female`, `male`, `ambiguous` |
| `manner_of_death` | string or null | `disease`, `accident`, `self_harm`, `assault`, `pending_investigation`, `unknown` |
| `hiv_status` | string or null | `negative`, `positive`, `exposed`, `unknown` |
| `birth_weight_g` | number or null | Birth weight in grams |
| `weight_kg_admission` | number or null | Weight at admission in kg |
| `weight_kg_death` | number or null | Weight at death in kg |
| `muac_cm_admission` | number or null | Mid-upper arm circumference at admission in cm |
| `muac_cm_death` | number or null | Mid-upper arm circumference at death in cm |

**User**

| Field | Type | Description |
|---|---|---|
| `name` | string | Name entered in the app, may be empty |
| `id` | string | User ID entered in the app, may be empty |

### chunks.jsonl

One line per piece of transcript.

| Field | Type | Description |
|---|---|---|
| `chunk_id` | string | Chunk ID, `imported` for a transcript brought in from elsewhere |
| `segment` | integer | Recording the chunk belongs to, from 0 |
| `timestamp` | Unix time | When the audio of the chunk was captured |
| `received_at` | Unix time | When the chunk was stored |
| `text` | string | English text |
| `annotations` | list of annotation | Medical terms found in `text` |
| `original_text` | string | Text in the spoken language, when it isn't English |
| `original_language` | string | Code of the spoken language, with `original_text` |
| `imported` | boolean | Present and true for an imported transcript |

**Annotation** (the Gilda annotation format)

| Field | Type | Description |
|---|---|---|
| `text` | string | Matched span of `text` |
| `start`, `end` | integer | Character offsets of the span |
| `matches` | list | Candidate terms, best first. Each has `term` (`db`, `id`, `entry_name`, `text`, `norm_text`, `status`, `source`), `score` (0 to 1), `url` and `match` details |

### edits.jsonl

One line per transcript correction.

| Field | Type | Description |
|---|---|---|
| `edit_id` | string | Edit ID, repeated in the inference it led to |
| `through` | string | Last chunk ID covered by the correction |
| `edited_at` | Unix time | When the correction was saved |
| `live` | boolean | Whether it was made while recording |
| `before` | string | Transcript text before the correction |
| `after` | string | Transcript text after the correction |
| `annotations` | list of annotation | Medical terms found in `after` |
| `user` | user | Person who made the correction |

### inference.jsonl

One line per inference result shown on screen.

| Field | Type | Description |
|---|---|---|
| `shown_at` | Unix time | When the result was shown |
| `chunk_id` | string | Last chunk the result covers |
| `timestamp` | Unix time | Capture time of that chunk |
| `chunks_processed` | integer | Number of chunks the result is based on |
| `source` | string | `recording` or `clinical_note` |
| `causes` | object | Ranked causes of death, keyed by code, for example `icd11:1F40`. Each has `name`, `identifiers` (for example `{"icd11": "1F40"}`) and `score`, a probability. Scores sum to 1 |
| `reasoning` | string or null | Explanation given with the result, or the error when inference failed |
| `questions` | list of string or null | Follow-up questions shown |
| `questions_done` | boolean or null | Whether question selection would stop asking |
| `chain` | list or null | Causal chain from the top underlying cause to death, each item with `name` and `identifiers` |
| `edit_id` | string | Present when the result followed a transcript correction |
| `notes` | list | Present for `clinical_note` results: `filename` of each note, and its `text` when `store.notes` is on |
| `metadata` | metadata | Case ID and profile when the result was made |

### timing.jsonl

One line per chunk and per inference, in seconds.

| Field | Type | Description |
|---|---|---|
| `kind` | string | `chunk` or `inference` |
| `chunk_id` | string | Chunk ID |
| `at` | Unix time | When the chunk was stored or the result shown |
| `translate_s` | number | Translation time (chunks, when translated) |
| `ground_s` | number | Grounding time (chunks) |
| `inference_s` | number | Time spent by the inference agent (inferences) |
| `request_s` | number | Total inference request time (inferences during recording) |
| `buffer_wait_s` | number | Time the chunk waited before inference started (inferences during recording) |

### revisions.jsonl

One line per change to the case: a recording, an inference from notes, a
transcript correction, a new inference or a profile change.

| Field | Type | Description |
|---|---|---|
| `revision` | integer | Revision number, from 1 |
| `kind` | string | `recording`, `notes`, `edit`, `inference` or `metadata` |
| `at` | ISO time | When the revision was saved |
| `user` | user | Person who made the change |
| `metadata` | object | Present when the case ID or profile changed: `before` and `after`, each a metadata object or null |

### last_case_number

A text file holding one integer, the highest `CODA-NNN` case number issued on
this laptop. The app suggests the next number for a new case.

## Application log

The log records the services starting and stopping, model loading, which
case folder each recording goes to, errors, and warnings. On a Mac install
both CODA services append to `coda.log`, which is never rotated or cleared.
Each line starts with the level and time, for example
`INFO: [2026-11-12 08:54:13] uvicorn.error - Started server process`.

At the default level, `INFO`, the log holds no transcript text. At `DEBUG`
(`logging.level`), it also holds the transcript text and top cause of each
inference.

## Data kept in the browser

Separately from the case folders, each browser can keep its own copy of its
sessions when "Save sessions on this device" is on in Settings. This copy
holds the transcript and results, without audio or note text, in the
browser's local storage on that device. It also remembers the user's name and
ID, the UI language and the last cohort picked.
