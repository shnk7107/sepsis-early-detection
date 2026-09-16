# Results — canonical, corrected

These two tables are the **reportable outputs**. They supersede the raw per-stage
significance files (`04_ablation_grid/`, `05_panel_decomposition/`, `06_significance_testing/`,
`07_crossmodel_validation/`), which are retained as **derivation/provenance** only.

## `RESULTS_primary_RF_1none.csv`
Primary analysis: Random Forest, no resampling (`1_none`), Nadeau–Bengio corrected,
Benjamini–Hochberg over the 3 arms.

| Channel | ΔAUPRC | p (NB) | p (BH/3) | sig |
|---|---|---|---|---|
| total_labs | 0.0216 | 0.013 | 0.038 | yes |
| SOFA_system | 0.0090 | 0.060 | 0.091 | no |
| ordering_beyond_value | 0.0050 | 0.330 | 0.330 | no |

**Read:** the lab signal is carried by values, not ordering.

## `RESULTS_robustness_sweep.csv`
Effect sizes + NB p-values across 3 models × 3 arms × 8 resampling conditions
(**72 rows**, deduplicated). Reported as a robustness sweep, **not** an N/24 significance
count — the multiplicity family is not well-defined across correlated resampling variants,
so absolute significance counts are not reported here. The invariant claim is the
asymmetry (physiology > ordering), which holds under every correction.

## Why the raw per-stage files are not the report
`day5`/`day6`/`day7` applied Benjamini–Hochberg over different scopes (40, 24, 72 tests),
so the same underlying test could be called significant in one file and not in another.
That inconsistency is resolved by reporting only from the two files here.
See `../FINDINGS.md` for the full corrected statement.

## `RESULTS_channel_reconstruction.csv`

Standalone-vs-marginal reconstruction for the lab-ordering and lab-value channels.
One row per design × site × model (condition `1_none`, 25 repeated-CV splits per cell).

Columns: `prevalence_floor` (AUPRC no-skill baseline = positive prevalence),
`full_auprc` / `ordering_only_auprc` / `values_only_auprc` with SDs (the three
absolutes — report these), `ordering_raw_ratio_pct` (ratio to zero baseline —
**do not report**, retained only to show what the naive figure would be), and
`ordering_above_baseline_pct` = (ordering_only − prevalence)/(full − prevalence).

The above-baseline fraction is **site-dependent (23 %–66 %)**; site A recovers
46–66 %, site B 23–31 %, with model choice mattering little. Quote the range, not
a point estimate. See FINDINGS.md § "Standalone channel reconstruction".
