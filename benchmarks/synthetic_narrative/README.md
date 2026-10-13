# Synthetic narrative benchmark

Measures how much text CODA's live transcriber emits while nobody is speaking,
and what suppressing it costs on the speech. Synthetic readings of the
lang_support narratives are spliced with silence and noise; each recording is
played through the `whisper-livekit` transcriber, built through
`create_transcriber`, so the `vac` and `decode_options` settings in
`config/settings.yaml` are what is under test.

Run everything from the repo root, in an environment with CODA installed.

## 1. Generate the data

```bash
python benchmarks/synthetic_narrative/audio_bank.py
python benchmarks/synthetic_narrative/non_speech_bank.py
python benchmarks/synthetic_narrative/splice.py
```

The first renders the seven narratives of each language (en, pt_br, bn, ts)
with Meta's MMS-TTS, about 600 MB of models on first use. The second builds
twelve non-speech clips: six synthesized, six real recordings from the DEMAND
corpus, a 560 MB download of which 25 MB is kept. The third writes 56
recordings under `data/benchmark/<variant>/<language>/`, each with a JSON map
of its speech and noise regions. The result is identical for the same seed.

## 2. Run the configurations

```bash
python benchmarks/synthetic_narrative/compare_configurations.py --languages en
```

This runs every configuration in the table at the top of
`compare_configurations.py` on the English recordings, one after another,
writes `results/lockstep/<config>/<variant>/<language>/<case>.json` with a
`.score.json` beside it, and prints the summary when done: one row per
configuration, variant and language with gap words per minute of non-speech,
per noise clip and in total, speech WER, and clock time over audio time.
Narrow it with `--configs`, `--variants` and `--languages`. A recording that
already has a complete result is skipped, so a stopped run resumes.
`--table` reprints the summary without running anything.

By default audio is fed in lock-step, each piece after the previous one is
processed, so the result does not depend on the machine's speed. Add
`--realtime` to feed at live speed instead and measure whether this machine
keeps up; run that with the machine otherwise idle.

Tsonga recordings are generated but cannot be run with the stock Whisper
models, which have no Tsonga.
