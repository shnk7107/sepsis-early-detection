# Data

The pipeline operates on the **PhysioNet/Computing in Cardiology Challenge 2019**
sepsis dataset (two ICU cohorts, ~40k patients). Raw data is not redistributed here.

- Source: https://physionet.org/content/challenge-2019/
- The label used is the challenge-provided `SepsisLabel` (Sepsis-3, 6 h pre-onset).

## Generated binaries (git-ignored — rebuild by running stages in order)

| File | Produced by | Stage |
|---|---|---|
| `combined.parquet` | `sepsis_loader.py` | 01 |
| `septic_windows.parquet`, `control_windows.parquet` | `sepsis_window.py`, `sepsis_match.py` | 02 |
| `features.parquet`, `feature_tags.csv` | `sepsis_aggregate.py` | 02 |
| `*crosssite_preds*.csv` | PS_A run scripts | 08 |

`feature_tags.csv` is small and IS tracked (it drives every ablation).

## Bundled copies

The binaries above are also included, deduplicated, under `data/` — see `data/README.md`
for a per-file description. `data/source/` holds the dataset and pipeline intermediates;
`data/predictions/` holds the regenerable PS-A cross-site prediction dumps.
