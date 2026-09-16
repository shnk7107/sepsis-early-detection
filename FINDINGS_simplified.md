# Findings — Sepsis Ordering vs Physiology Study (Simplified)

## Research Question
We wanted to determine how much of a sepsis model's performance comes from:
- clinician ordering behaviour (whether and when a lab test was ordered), and
- patient physiology (the actual lab values).

We also wanted to know whether standard feature ablation can correctly separate these two sources of information after proper statistical correction.

## Main Finding
The model relies mainly on **laboratory values**, not on **laboratory ordering behaviour**.

After applying proper statistical corrections:
- the ordering channel was **not statistically significant**,
- while the overall laboratory signal remained significant.

## Main Results
- Removing all laboratory information caused a small but significant drop in performance.
- Removing only ordering information caused a very small drop and was not statistically significant.
- Removing the complete SOFA system was also not significant after correction.

## Why Statistical Correction Matters
Without correction:
- Ordering appeared significant in 8/8 experiments.

After Nadeau–Bengio:
- Only 1/8 remained significant.

After Benjamini–Hochberg:
- 0/8 remained significant.

The laboratory-value signal remained much stronger, showing that the corrections removed weak effects while preserving stronger ones.

## Circularity Check
The PhysioNet SepsisLabel is not based only on SOFA values. It also depends on infection-suspicion timing.

Removing all SOFA variables did not eliminate the main physiological signal, showing that the model is not simply reproducing the label definition.

## Redundancy
Removing the ordering channel caused only a small performance drop.

However, when trained by itself, the ordering channel still recovered a meaningful amount of predictive performance.

This shows that physiology and ordering contain overlapping information.

## Contribution
This work does not introduce a new algorithm.

Its contribution is a statistically rigorous evaluation of clinician-ordering and physiological information in sepsis prediction using multiple robustness experiments.

## Limitations
- Two hospitals only.
- Limited statistical power.
- Ordering effects were mostly not statistically distinguishable from zero.
- Results are specific to this dataset and design.

## Important Pipeline Decisions
- Leakage was prevented using only the six hours before sepsis onset.
- Nadeau–Bengio corrected repeated cross-validation statistics.
- Benjamini–Hochberg controlled false discoveries.
- Controls were matched by hospital and onset position.
