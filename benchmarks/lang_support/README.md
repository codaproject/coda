# Language support benchmark

Datasets are keyed by `bn` for Bengali, `pt_br` for Brazilian Portuguese,
`ts` for Tsonga, the language also known as Shangaan, `zu` for Zulu, `st` for
Sesotho, and `syl` for Sylheti. Each contains the same seven recorded COD case
narratives. Bengali has two independent recording sets of them.

```
lang_support/
  data/
    bn/
      audio/          # original recording filenames; local, ignored by Git
        set1/         # one directory per recording set
        set2/
      references/     # original JSON files, unchanged
      originals/
    pt_br/
      audio/
      references/
      originals/      # original delivery archive; local, ignored by Git
    ts/
      audio/
      references/
    zu/
      audio/
      references/     # built from the supplied translation document
      originals/      # supplied translation document; local, ignored by Git
    st/
      audio/
      references/
      originals/
    syl/
      audio/
      originals/
  languages/          # adapters for supplied filenames and JSON quirks
  results/
    bn/               # existing results; local, ignored by Git
    pt_br/
```

`dataset_io.load_samples(language)` returns samples with `case_id`, `audio_path`,
`reference`, and `recording_set`. Bengali has two recording sets, each read
from its own translation of the cases and scored against it. `set1` uses
`cases_bn_filtered.json`, and the full `cases_bn.json` is retained as supplied.
`set2` uses `cases_bn_set2.json`, converted from the supplied Bijoy-encoded
text, with three words that were lost in delivery restored and each
slash-separated alternative, such as র্যাশ/ফুসকুড়ি, reduced to the one the
reader spoke. Means are taken over both sets and also reported per set.
Sylheti is scored
with the Bengali engines and normalization. Until its references arrive as
`references/cases_syl.json`, its recordings are transcribed but not scored.
The Portuguese adapter handles `Iri_*`
filenames and Portuguese text stored under the stale `bn_narrative` field.
The Zulu and Sesotho recordings arrived in one batch with inconsistent
filenames, so their adapters map each supplied filename to its case. An adapter
can mark data artifacts. `MISMATCHED` recordings were read from a different
translation than the reference and are left unscored (Zulu `lri_1`).
`SPOKEN_TITLE` recordings open with a title that is not in the reference, and
words before the first reference word are not counted (Zulu `lri_2`,
`diarrhea_1`). After changing references or these markers, rescore saved
transcripts with `run_benchmark.py --language <tag> --rescore`.
Duplicate cases, unmatched audio, conflicting references, or missing recordings
produce explicit errors. Place the original recordings under the corresponding
`audio/` directory before running; recordings are not committed to Git.

The existing runners remain available during consolidation. From the repo root:

```bash
PYTHONPATH="$PYTHONPATH:benchmarks/lang_support" python benchmarks/lang_support/run_benchmark.py --language bn indic-conformer
PYTHONPATH="$PYTHONPATH:benchmarks/lang_support" python benchmarks/lang_support/run_benchmark.py --language pt_br whisper-small
PYTHONPATH="$PYTHONPATH:benchmarks/lang_support" python benchmarks/lang_support/run_benchmark.py --language ts mms-1b-all
PYTHONPATH="$PYTHONPATH:benchmarks/lang_support" python benchmarks/lang_support/run_benchmark.py --language zu anv-zul-turbo
PYTHONPATH="$PYTHONPATH:benchmarks/lang_support" python benchmarks/lang_support/run_benchmark.py --language st anv-sot-turbo
PYTHONPATH="$PYTHONPATH:benchmarks/lang_support:src" python benchmarks/lang_support/run_benchmark.py --language syl indic-conformer
PYTHONPATH="$PYTHONPATH:src" python benchmarks/lang_support/bn_asr_bench.py --help
python benchmarks/lang_support/pt_br_asr_bench.py --help
python benchmarks/lang_support/plot_engines.py --results_dir benchmarks/lang_support/results/bn
```

Results under `results/<language>/` were produced in one pass by the current
pipeline, scoring both languages over the same seven cases. Each file records the
host hardware, the faster-whisper compute type, and any clips that failed.

Engines that reach a rate-limited API are retried, and a clip that never returns
text is recorded under `failed` rather than scored as a complete
mis-transcription. Only the successful attempt is timed, so a retry does not
inflate RTF.

`indic-whisper` is a Hindi fine-tune kept as a cross-script control. It
recognizes Bengali speech but writes it in Devanagari, so its word error rate
exceeds 1.0 by design and should not be read as a broken engine.
