# Findings — Sepsis ordering-vs-physiology ablation study

*Canonical, corrected statement of results. Every number here traces to a verified
output file (RESULTS_primary_RF_1none.csv, RESULTS_robustness_sweep.csv [deduped, 72 rows],
day6 significance, day7 cross-model). Where an earlier draft used different framing,
this document supersedes it.*

## Research question

On PhysioNet 2019, how much of a sepsis model's performance is carried by clinician
**ordering behavior** (whether/when a lab was drawn) versus **physiology** (the measured
values), and can standard feature ablation even distinguish the two once cross-validation
variance is corrected for?

## Headline (state it as the relative claim)

The lab signal is carried by **values, not ordering**. At the primary specification, the
total lab contribution is a small but significant AUPRC drop when removed; the
**ordering-beyond-value** contribution is **not distinguishable from zero** under correction.
The direction (physiology > ordering) is invariant to correction method; only the magnitude
of the positive claim moves.

> Do **not** state this as a "9:1" ratio. The reported tables do not produce a 9:1
> decomposition; that number is retired.

## Primary analysis — RF, no resampling (1_none), NB-corrected, BH over 3 tests

| Channel | mean ΔAUPRC | p (NB) | p (BH/3) | significant |
|---|---|---|---|---|
| total_labs (drop all labs) | 0.0216 | 0.0127 | **0.038** | yes |
| SOFA_system (drop 7 SOFA vars, both channels) | 0.0090 | 0.0604 | 0.091 | no |
| ordering_beyond_value (drop lab ordering, keep values) | 0.0050 | 0.3302 | 0.330 | no |

Interpretation: total labs contribute a modest, significant amount; neither the
SOFA-system-specific drop nor the ordering-beyond-value drop is individually
distinguishable from zero. The lab signal sits in the values.

## The correction collapse (why correcting statistics matters) — RF, 8 conditions

| Channel | naive | NB-corrected | +BH (within) | +BH (global) |
|---|---|---|---|---|
| ordering_beyond_value | 8/8 | 1/8 | 0/8 | 0/8 |
| SOFA_system | 8/8 | 2/8 | 1/8 | 1/8 |
| total_labs | 8/8 | 7/8 | 7/8 | 3/8 |

The ordering channel is significant in **8/8 conditions naively, 1/8 after NB, 0/8 after BH**
(consistent denominators, one model). Reported cross-model (3 models × 8 conditions): ordering
= **1/24 NB, 0/24 BH**; total labs = 21/24 NB and 21/24 under within-arm BH, attenuating to
2/24 under a global BH across all 72 tests (the two survivors are the variance-deflated
oversampling conditions). Report the single-model collapse for the clean denominator story;
cite cross-model as confirmation it is not RF-specific.

## Circularity check (reviewer Point 3) — answered

- The Sepsis-3 label used here is PhysioNet's precomputed `SepsisLabel`, which also requires
  infection-suspicion timing (antibiotics + cultures). It is **not a pure function of the SOFA
  feature values**, so the circularity is partial, not definitional.
- Dropping the full SOFA system (all_7, both channels) is not significant after correction
  (0/8 BH single-model, 0/24 BH cross-model). The result does not hinge on the label-defining
  variables.
- Caveat: non-SOFA labs correlate with the dropped SOFA labs (e.g. BUN↔Creatinine ~0.97), so
  the surviving signal is not *independent of* the SOFA organ systems — but it is still
  physiology, not ordering. Claim "not the literal label-defining columns," not "independent
  physiology."

## Standalone channel reconstruction (the redundancy demonstration)

Ablation and standalone performance disagree, and that disagreement is the point. Dropping
the lab-ordering channel costs a *marginal* ~0.005 AUPRC (not distinguishable from zero after
NB+BH). Yet the ordering channel **trained alone** recovers a substantial share of the model's
above-chance performance. Both are true: the value channel covers for ordering under ablation.

**Report the three absolute numbers, not a single percentage.** AUPRC's no-skill floor is the
positive prevalence, not zero, so any normalised ratio depends on a denominator choice a
reviewer can argue with. The absolutes cannot be argued with.

RF, condition `1_none`, 25 repeated-CV splits per cell. Full table (3 models × 2 sites ×
2 designs): [`results/RESULTS_channel_reconstruction.csv`](results/RESULTS_channel_reconstruction.csv).

| design | site | prevalence floor | full AUPRC | ordering-only AUPRC | above-baseline |
|---|---|---|---|---|---|
| matched | A | 0.0714 | 0.1698 | 0.1338 | 63.4 % |
| matched | B | 0.0714 | 0.2152 | 0.1041 | 22.7 % |
| native  | A | 0.0909 | 0.2190 | 0.1553 | 50.3 % |
| native  | B | 0.0556 | 0.1634 | 0.0817 | 24.2 % |

above-baseline = (ordering_only − prevalence) / (full − prevalence)

**The fraction is strongly site-dependent — do not headline a single number.** Across all
three model families it spans **23 %–66 %**: site A recovers 46–66 %, site B only 23–31 %.
Model choice barely matters (RF/LightGBM/XGBoost agree to within a few points inside a site);
*site* is what moves it. A pooled raw ratio (~64 %) hides this by averaging over heterogeneous
full-model performance and, in the native design, different prevalences. Quote the range and
the site split, and treat the site-dependence as a result in its own right rather than noise.

**Framing — this is a redundancy demonstration, not a new method.** That leave-one-out
ablation under-weights a feature redundant with a retained one is documented (Strobl 2008 on
correlated-predictor importance bias; it is the standard motivation for Shapley-based
importance). The contribution here is the *domain-specific empirical characterisation* in
sepsis EHR data — quantifying how far apart the marginal and standalone readings sit for the
clinician-ordering channel specifically. Do not call it "ablation blindness" or present it as
a novel diagnostic; a reviewer will correctly say the limitation is already known.

## On the "physiology = panel − indicators" decomposition

Report the value-carried signal as a **value contribution**, not "physiology," and flag it as
**approximate/non-additive**: ablation deltas do not sum under redundancy (the project's own
central finding), so `total_lab = value + ordering` does not hold as an identity. Present the
three arms as separate contrasts, not additive components.

## Relationship to prior work

A recent **pre-registered, four-dataset falsification study** (Dickens-class) asks the same
question and finds physiology dominates. This work **corroborates** that finding with
**NB + BH corrected significance testing** on the ablation deltas and adds a **methodological
critique**: standard ablation under-reports a redundant channel (the ordering channel alone
reconstructs most of the model, yet its marginal drop is ~2% and non-significant). Novelty is
modest and corroborative; the corrected-significance collapse and the ablation-blindness
demonstration in the AUPRC regime are the contribution.

## Limitations (state explicitly)

1. **Cohort scope:** onset-position cutoff C=48 restricts to early-to-mid-stay onset; late-onset
   sepsis is out of scope. Report how many septic patients this excludes.
2. **Cross-site (former "Finding 5") is a limitation, not a result.** The two PhysioNet sites are
   similar; failing to see worse cross-site ordering-reliance transport is an **underpowered
   two-site null**, not a non-replication. The Dickens-class study found a care-intensity signal
   at multi-center (many hospitals) that a two-site design cannot rule in or out.
3. **Power:** 5-fold (repeated 5×5) CV with NB correction has low power against effects of
   0.005–0.02 AUPRC. Non-significant results mean "not distinguishable from zero," never "zero."
4. **The positive claim is modest:** total_labs clears BH at 0.038, the lone primary survivor,
   effect size ~0.022 AUPRC.
5. **Attribution/family dependence:** absolute significance counts depend on the multiplicity
   family; the invariant claim is the asymmetry (physiology > ordering), reported via primary +
   effect-size sweep rather than N/24 counts.

## Verified-correct pipeline components

- **Leakage guard:** features drawn only from `[onset − 6h, onset − 1]`, strictly before the
  first positive label; leakage removed, residual pre-onset SOFA signal is what the ablation
  measures.
- **NB correction:** `t = mean(d) / sqrt((1/J + ρ)·var(d))`, ρ = 1/(k−1) = 0.25 for 5-fold;
  Day 6 uses repeated 5×5 CV for a stable variance estimate.
- **all_7:** drops both value and indicator channels of all seven SOFA vars (full SOFA-system
  drop).
- **Control matching:** on onset-position (C=48) and site, closing those leakage paths.
