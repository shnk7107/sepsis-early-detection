# Data binaries

This folder holds every large binary the pipeline uses, **deduplicated to one copy each**
(the original repo carried these copied across many stage folders — e.g. `features.parquet`
appeared five times). They are kept here, separate from the code, so each pipeline stage stays
readable. Scripts in `01_…08_` originally read these from their own working directory; point
them here (or copy the needed file next to the script) when re-running.

## `source/` — dataset and pipeline intermediates

| File | Size | Produced by | Consumed by | What it is |
|---|---|---|---|---|
| `combined.parquet` | 15.7 MB | `01_data_loading/sepsis_loader.py` | stage 02 | All PhysioNet 2019 records merged into one hourly table (vitals, labs, demographics, `SepsisLabel`, onset). |
| `septic_windows.parquet` | 0.1 MB | `02_…/sepsis_window.py` | stage 02 | Hourly rows for septic patients, restricted to the 6 h pre-onset window (leakage-free). |
| `control_windows.parquet` | 1.3 MB | `02_…/sepsis_match.py` | stage 02 | Non-septic control windows, matched to cases on site and onset-position. |
| `features.parquet` | 2.1 MB | `02_…/sepsis_aggregate.py` | **stages 04–07** | Per-patient aggregated feature matrix (243 features = value + indicator channels × summary statistics) with label. **This is the primary input to every ablation.** |
| `features_native.parquet` | 2.0 MB | PS-A native run | stage 08 | Same aggregation for the native-prevalence cross-site cohort. |
| `control_windows_native.parquet` | 1.3 MB | PS-A native run | stage 08 | Native-prevalence control windows for the cross-site analysis. |

`feature_tags.csv` (the SOFA/channel flags that drive every ablation) is **small and tracked in
the stage folders**, not here.

## `predictions/` — regenerable cross-site prediction dumps

Per-patient predicted probabilities from the PS-A cross-site experiments (train on one site,
test on the other). Used only to compute the transportability metrics; **not read by any other
stage**. Fully regenerable by re-running the stage 08 scripts. Kept for completeness.

| File | Size | Origin |
|---|---|---|
| `psA_matched_base_preds.csv` | 10.6 MB | Matched-prevalence run |
| `psA_matched_enhanced_preds.csv` | 20.9 MB | Matched-prevalence, enhanced |
| `psA_matched_mirror_preds.csv` | 26.8 MB | Mirror-arm patch |
| `psA_native_base_preds.csv` | 10.3 MB | Native-prevalence run |
| `psA_native_enhanced_preds.csv` | 20.3 MB | Native-prevalence, enhanced |
| `psA_native_mirror_preds.csv` | 26.0 MB | Native mirror-arm patch |

## Note on version control

These binaries total ~138 MB. If you commit them, the `.gitignore` has been set to allow
`data/**` while still blocking stray copies elsewhere in the tree. For a lean GitHub repo you
may prefer to keep `data/` local and regenerate from PhysioNet (see `../DATA.md`) — the
prediction dumps especially are large and regenerable.
