# References

Prior-art and methodological references for this study. Every entry below was
located in an indexed source (PubMed, Oxford Academic, medRxiv, Preprints.org,
arXiv, or a publisher page) and verified by opening it. Verify each DOI/link
independently before citing in a submission.

## Nearest neighbor (same question, same task)

- **Dickens, A. (2026).** *Falsification Testing of Sepsis Prediction Models:
  Evaluating Independent Biological Signal After Controlling for Care-Process
  Intensity.* medRxiv 2026.03.17.26348414.
  Pre-registered (OSF), four datasets including PhysioNet 2019; finds physiology
  dominates care-process intensity. **This work corroborates Dickens with NB+BH
  corrected significance and an ablation-audit critique.** Uses AUROC and
  pre-specified thresholds; no CV-variance correction.

## Ordering behavior / informative missingness (Finding 1 — cited, not claimed)

- **Agniel, D., Kohane, I. S., & Weber, G. M. (2018).** Biases in electronic
  health record data due to processes within the healthcare system. *BMJ*,
  361:k1479. PMID 29712648.
- **Singh, H., Sato, R., & Ohkuma, T. (2021).** On missingness features in
  machine learning models for critical care. *JMIR Medical Informatics*.
  PMID 34889756. *(peer-reviewed anchor for the ordering-signal result)*
- **Yu, Y. (2026).** What the Doctor Didn't Order: Informative Missingness as a
  Clinical Signal in ICU Prediction Models. Preprints.org
  10.20944/preprints202606.1130. *(preprint; secondary support — missingness
  rebuilds 84–93% of value-only performance on MIMIC-IV)*
- **Che, Z., et al. (2016).** Recurrent Neural Networks for Multivariate Time
  Series with Missing Values (GRU-D). OpenReview / *Scientific Reports* 2018.
  *(canonical informative-missingness reference)*
- **Sisk, R., et al. (2021).** Informative presence and observation in routine
  health data. *JAMIA*.
- **Goldstein, B. A., et al. (2019).** Informative visit / observation processes
  in EHR modeling.
- **Tan, Y., et al. (2023).** Informative missingness in clinical prediction.

## Ablation under-reports redundant features (the mechanism — cited, not claimed)

- **Strobl, C., et al. (2008).** Conditional variable importance for random
  forests. *BMC Bioinformatics*. *(collinearity distorts importance)*
- **UMFI — Unbiased Measures of Feature Importance.** arXiv 2204.09938.
- **Piórkowska, N., et al. (2026).** Interpretability as stability under
  perturbation reveals systematic inconsistencies in feature attribution.
  medRxiv 2026.04.20.26351354. *(attribution ≠ functional importance; "latent
  dependence")*
- **Sridhar, S., Xue, A., & Wong, E. (2026).** Missingness Bias Calibration in
  Feature Attribution Explanations (MCal). arXiv. *(ablation feeds OOD inputs →
  unreliable importance; mechanism behind the audit critique)*

## Significance / cross-validation variance (the correction — the contribution)

- **Nadeau, C., & Bengio, Y. (2003).** Inference for the generalization error.
  *Machine Learning*, 52. *(the corrected resampled t-test used throughout)*
- **Altmann, A., et al. (2010).** Permutation importance: a corrected feature
  importance measure. *Bioinformatics*, 26(10):1340. PMID 20385727.
- **Benjamini, Y., & Hochberg, Y. (1995).** Controlling the false discovery
  rate. *JRSS-B*, 57(1):289–300. *(the multiplicity correction)*
- **Riezler, S., & Hagmann, M. (2024).** *Validity, Reliability, and
  Significance: Empirical Methods for NLP and Data Science*, 2nd ed. Springer,
  ISBN 978-3-031-57064-3. *(GAM validity test for circular features; LMEM
  significance test — nearest neighbor on method)*
- **Hagmann, M., & Riezler, S. (2021).** False perfection in machine
  prediction: detecting and assessing circularity problems in machine learning.
  arXiv 2106.12417. *(canonical circularity paper; uses SOFA/sepsis case)*

## Class imbalance / resampling harm (SMOTENC result — cited)

- **van den Goorbergh, R., et al. (2022).** The harm of class imbalance
  corrections for risk prediction models. *JAMIA*, 29(9):1525–1534.

## Cross-site / generalization (Finding 5 — now a limitation)

- **Yamamoto, R., Wu, F., Sprehe, L. K., Abeer, A., Celi, L. A., & Tohyama, T.
  (2026).** Observation-process features are associated with larger domain shift
  in sepsis mortality prediction: MIMIC-IV vs eICU-CRD. medRxiv
  2026.04.05.26350209. *(care-intensity signal appears multi-center, absent
  single-site — the reason our two-site null is a limitation)*
- **Futoma, J., et al. (2021).** Generalization in clinical prediction models:
  the blessing and curse of measurement indicator variables. *Critical Care
  Explorations*, 3(7):e0453. PMID 34235453.
- **Ehlers, S. F., Tranchellini, F., et al. (2025).** Case-Control Matching
  Erodes Feature Discriminability for AI-driven Sepsis Prediction. medRxiv
  2025.06.26.25330281. *(justifies the matched-vs-native two-design check)*

## Cross-domain circularity (analogous problem, other diseases)

- **Xu, J., & Costen, F. (2026).** Machine-learning multiclass classification of
  cognitive stages... label circularity in Alzheimer's staging. *Diagnostics*,
  16(12):1755. 10.3390/diagnostics16121755.
- **Verma, K., & Kumar, S. (2025).** Design-Induced Circularity in Alzheimer's
  Biomarker Trials. medRxiv 2025.12.20.25342753. *(circularity inflates FPR to
  0.994 under null vs 0.047 with proper design)*

## Uncorrected-ablation-audit example (the target of the critique)

- **Ristori, M. V., et al. (2026).** Machine Learning Models for Sepsis: From
  Early Detection to Short- and Long-Term Prognosis. *Int. J. Mol. Sci.*,
  27(6):2721. 10.3390/ijms27062721. *(ablation-based circularity check on a
  single split, no CV-variance correction — the practice this work critiques)*
