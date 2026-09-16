# Sepsis prediction: ordering behavior vs. physiology

A falsification-style ablation study on **PhysioNet 2019**. The question: how much of a
sepsis model's performance comes from clinician **ordering behavior** (whether/when a lab
was drawn) versus real **physiology** (the measured values) — and can standard feature
ablation even tell them apart once cross-validation variance is corrected for?

**Headline result.** The lab signal is carried by **values, not ordering**. The
ordering-beyond-value contribution is significant in 8/8 conditions under naive testing,
but collapses to 0/8 once Nadeau–Bengio and Benjamini–Hochberg corrections are applied. The
total lab contribution survives correction; the ordering channel does not. See
[`FINDINGS.md`](FINDINGS.md) for the full, corrected statement and limitations.

> **Status:** experiments complete; analysis corrected and frozen. Reportable numbers live
> in [`results/`](results/). Per-stage significance files are retained as provenance and are
> superseded by `results/` — see [`results/README.md`](results/README.md).

## Pipeline (run in order)

| Stage | Folder | What it does |
|---|---|---|
| 01 | `01_data_loading/` | Load PhysioNet 2019, patient-level labels + onset (`sepsis_loader.py`) |
| 02 | `02_windowing_and_matching/` | 6 h pre-onset windows, site/position-matched controls, feature aggregation + SOFA tagging |
| 03 | `03_train_test_split/` | Frozen 80/20 split, stratified by label × site (`day3_test_pids.json`) |
| 04 | `04_ablation_grid/` | AUPRC ablation grid × 8 resampling conditions; supporting perm-importance / paired tests |
| 05 | `05_panel_decomposition/` | Panel-level ablation: total-lab / value / ordering decomposition |
| 06 | `06_significance_testing/` | Nadeau–Bengio corrected significance (repeated 5×5 CV) + naive baseline |
| 07 | `07_crossmodel_validation/` | Repeats significance across RF / XGBoost / LightGBM |
| 08 | `08_crosssite_analysis/` | PS-A: train-on-one-site / test-on-other, matched & native prevalence |

## Key design decisions (verified)

- **No label leakage.** Features come only from `[onset − 6 h, onset − 1]`, strictly before
  the first positive label. The residual pre-onset SOFA signal is what the ablation measures.
- **Correct variance correction.** Nadeau–Bengio corrected resampled *t*-test
  (`ρ = 1/(k−1) = 0.25` for 5-fold), on repeated 5×5 CV.
- **Full SOFA-system ablation** (`all_7`) drops both the value and the ordering channel of all
  seven SOFA variables present in PhysioNet.
- **Circularity is partial:** the label is the challenge-provided `SepsisLabel`, which also
  requires infection-suspicion timing, so it is not a pure function of the SOFA feature values.

## Reproduce

```bash
pip install -r requirements.txt
```

Obtain the PhysioNet 2019 data (see [`DATA.md`](DATA.md)), then run each stage's script in
numeric order. Large binaries (`*.parquet`, prediction dumps) are git-ignored and regenerated
by the pipeline; `feature_tags.csv` is tracked because it drives every ablation.

`run_all.py` is the pipeline manifest and auditor:

```bash
python run_all.py --list     # canonical stage order + authoritative script per stage
python run_all.py --check    # verify every declared output is present (17/17)
python run_all.py --analysis # re-run the local, analysis-only stages
```

It also records the seed policy (`SEED = 42` throughout) and the fold-internal
resampling argument (samplers are steps inside an `imblearn` Pipeline handed to
`cross_validate`, so they only ever see each fold's training portion; the test
PIDs are frozen in stage 03 before any fitting).

Where a stage has several script versions, `run_all.py --list` names the
authoritative one. In particular, the Stage 08 scripts under
`Mirror Arm Patch/Scripts/` supersede the earlier 2-arm and 5-arm variants —
they are **not** duplicates.

## Repository layout

```
├── README.md              ← this file
├── FINDINGS.md            ← corrected, canonical findings + limitations
├── REFERENCES.md          ← prior-art + methodological references (Dickens nearest neighbor)
├── DATA.md                ← data source and generated-binary manifest
├── requirements.txt       ← PIN VERSIONS before submission (see file header)
├── run_all.py             ← pipeline driver: --list / --check / --analysis
├── results/               ← REPORTABLE outputs (supersede per-stage files)
└── 01_… 08_…              ← pipeline stages, in order
```
