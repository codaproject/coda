# Language support benchmark

Datasets use BCP 47 tags: `bn` for Bengali and `pt-BR` for Brazilian Portuguese.
Each contains the same seven recorded COD case narratives.

```
lang_support/
  data/
    bn/
      audio/          # original recording filenames; local, ignored by Git
      references/     # original JSON files, unchanged
    pt-BR/
      audio/
      references/
      originals/      # original delivery archive; local, ignored by Git
  languages/          # adapters for supplied filenames and JSON quirks
  results/
    bn/               # existing results; local, ignored by Git
    pt-BR/
```

`dataset_io.load_samples(language)` returns samples with `case_id`, `audio_path`,
and `reference`. The Bengali adapter uses `cases_bn_filtered.json`; the full
`cases_bn.json` is retained as supplied. The Portuguese adapter handles `Iri_*`
filenames and Portuguese text stored under the stale `bn_narrative` field.
Duplicate cases, unmatched audio, conflicting references, or missing recordings
produce explicit errors. Place the original recordings under the corresponding
`audio/` directory before running; recordings are not committed to Git.

The existing runners remain available during consolidation. From the repo root:

```bash
PYTHONPATH="$PYTHONPATH:benchmarks/lang_support" python benchmarks/lang_support/run_benchmark.py --language bn indic-conformer
PYTHONPATH="$PYTHONPATH:benchmarks/lang_support" python benchmarks/lang_support/run_benchmark.py --language pt-BR whisper-small
PYTHONPATH="$PYTHONPATH:src" python benchmarks/lang_support/bangla_asr_bench.py --help
python benchmarks/lang_support/ptbr_asr_bench.py --help
python benchmarks/lang_support/plot_engines.py --results_dir benchmarks/lang_support/results/bn
```

Old result files are preserved without rescoring. Some Bengali files include an
unrelated eighth `bangla_test` sample; seven-case comparisons must exclude it.
The exploratory `bangla_test` files and download scripts remain in
`benchmarks/banglaspeech2text/` and are not part of these datasets.
