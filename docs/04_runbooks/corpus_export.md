# Runbook: labelled trajectory corpus (v1) export

Source of truth: `docs/03_preregistration/DETECTION_PROTOCOL.md` section 5 and
`docs/06_corpus/codebook.md`. Code: `finswarm.analysis.corpus_v1`, driven by `scripts/analyze.py
--corpus-v1`. The published corpus is the labelling sheet plus its metadata under `docs/06_corpus/`;
the **sealed answer key never enters the repository** (the code refuses a `--key-out` inside it,
and `.gitignore` ignores any `key_SEALED.csv`).

## 1. Sample the blind sheet (one command)

```bash
uv run python scripts/analyze.py --corpus-v1 sample \
    --campaigns-root results/campaigns \
    --d9-scores <path to the cached D9 judge scores, parquet or csv> \
    --key-out ../finswarm-sealed/key_SEALED.csv \
    --n-items 60 --sampler-seed 20261011
```

- Candidates: every `TR-H*` and `CA-H*` campaign under `--campaigns-root` (trunks, branches, ghosts,
  probes and CA of the Haiku lane), development half only; `--corpus-campaigns A,B` restricts them,
  `--corpus-lane` changes the lane.
- Windows follow the trend rule when `results/analysis/gates/gates.json` exists; the S2-punished
  trunks come from `results/analysis/detection/corpus.parquet` when it exists (`--gates-json`,
  `--detect-corpus` to point elsewhere).
- Outputs: `results/analysis/corpus_v1/labelling_sheet.csv` (template header, only `item_id` and
  `text` filled) and `sheet_meta.json`; the key at `--key-out`, outside the repo. The command prints
  the sha256 of the sheet and of the key: **paste both into `docs/06_corpus/SEAL.md`** (the
  seal) before anyone labels.

## 2. Label, validate, agreement

```bash
# labellers fill results/analysis/corpus_v1/labelling_sheet.csv (codebook columns only)
uv run python scripts/analyze.py --corpus-v1 validate --sheet results/analysis/corpus_v1/labelling_sheet.csv
uv run python scripts/analyze.py --corpus-v1 kappa --sheet results/analysis/corpus_v1/labelling_sheet.csv \
    --sealed-key ../finswarm-sealed/key_SEALED.csv
```

`validate` exits 1 on a malformed sheet (header, missing codes); `kappa` prints the agreement
against the sealed key (DETECTION_PROTOCOL section 5; organizers' bar 2.2). Calibration material:
`docs/06_corpus/calibration_sheet.csv` and `calibration_answers_READ_AFTER.csv`.

