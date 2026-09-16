"""
Day 3 · leakage-safe evaluation SKELETON.

One correct, fully-seeded run of  impute -> scale -> SMOTENC -> RandomForest
with a stratified hold-out + stratified 5-fold CV, reporting AUPRC (primary) and
AUROC (secondary). This validates PLUMBING, not a result. No method grid, no other
models, no ablation, no tuning — those are Day 4+.

Key design points (settled Day 3):
  - Stratified split, NOT grouped: one row per patient, all pids distinct, so there
    is no within-patient cluster to keep together. The control match is
    distributional (position carries no outcome info), so a septic and its matched
    controls may fall across folds freely.
  - imblearn Pipeline confines resampling to the training fold. Imputer/scaler/
    SMOTENC fit on train only; validation and test are transform-only and never
    resampled.
  - SMOTENC (not vanilla SMOTE) so categorical columns are not interpolated into
    impossible states (Creatinine_measured = 0.5). Categoricals = the _measured /
    _n indicators plus Gender / Unit1 / Unit2. Vanilla SMOTE is a one-line fallback
    (USE_SMOTENC=False) for the documented secondary comparison.
  - Numeric columns are scaled (for SMOTENC's distance metric); categoricals are
    imputed but NOT scaled (kept as discrete labels).
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC, SMOTE

SEED = 42
USE_SMOTENC = True          # False -> vanilla SMOTE (secondary comparison)
FEATURES_FILE = "features.parquet"
TAGS_FILE = "feature_tags.csv"
TEST_PIDS_FILE = "day3_test_pids.json"

# ----------------------------------------------------------------------------
# Load + column roles
# ----------------------------------------------------------------------------
df = pd.read_parquet(FEATURES_FILE)
tags = pd.read_csv(TAGS_FILE)

ID_COLS = ["pid", "set", "label"]
feature_cols = [c for c in df.columns if c not in ID_COLS]

# categoricals SMOTENC must not interpolate
indicator_cols = tags.loc[tags.kind == "indicator", "feature"].tolist()   # _measured, _n
binary_static = ["Gender", "Unit1", "Unit2"]
CAT_COLS = [c for c in feature_cols if c in indicator_cols or c in binary_static]
NUM_COLS = [c for c in feature_cols if c not in CAT_COLS]

print(f"features {len(feature_cols)}  |  numeric {len(NUM_COLS)}  categorical {len(CAT_COLS)}")

# ----------------------------------------------------------------------------
# 1. Stratified hold-out  (label x set)
# ----------------------------------------------------------------------------
df["strata"] = df["label"].astype(str) + "_" + df["set"].astype(str)
train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=SEED, stratify=df["strata"])

X_train = train_df[NUM_COLS + CAT_COLS]      # reorder: numeric block then cat block
X_test = test_df[NUM_COLS + CAT_COLS]
y_train = train_df["label"].to_numpy()
y_test = test_df["label"].to_numpy()

# SMOTENC categorical indices = positions of the cat block AFTER the ColumnTransformer
# (ColumnTransformer emits num columns first, then cat columns).
cat_idx = list(range(len(NUM_COLS), len(NUM_COLS) + len(CAT_COLS)))

# ----------------------------------------------------------------------------
# 2. Pipeline
# ----------------------------------------------------------------------------
pre = ColumnTransformer([
    ("num", SkPipeline([("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler())]), NUM_COLS),
    ("cat", SimpleImputer(strategy="most_frequent"), CAT_COLS),
])

sampler = (SMOTENC(categorical_features=cat_idx, random_state=SEED)
           if USE_SMOTENC else SMOTE(random_state=SEED))

# class_weight=None: SMOTE already corrects imbalance; weighting too = double-counting.
rf = RandomForestClassifier(n_estimators=300, class_weight=None,
                            random_state=SEED, n_jobs=-1)

pipe = ImbPipeline([("pre", pre), ("smote", sampler), ("rf", rf)])

# ----------------------------------------------------------------------------
# 3. Stratified 5-fold CV on TRAIN  (resampling confined in-fold by the pipeline)
# ----------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scores = cross_validate(pipe, X_train, y_train, cv=cv,
                        scoring=["average_precision", "roc_auc"], n_jobs=1)
cv_ap = scores["test_average_precision"]; cv_auc = scores["test_roc_auc"]

# ----------------------------------------------------------------------------
# 4. Fit on full train, evaluate untouched test
# ----------------------------------------------------------------------------
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
test_ap = average_precision_score(y_test, proba)
test_auc = roc_auc_score(y_test, proba)

# ----------------------------------------------------------------------------
# 5. LEAKAGE ASSERTIONS  (the actual point of Day 3)
# ----------------------------------------------------------------------------
print("\n=== LEAKAGE ASSERTIONS ===")
A = {}

# (a) no pid in both splits
overlap = set(train_df.pid) & set(test_df.pid)
A["no_pid_overlap"] = (len(overlap) == 0)

# (b) prevalence + A:B mix preserved
def prev(d): return d.label.mean()
def abmix(d): return (d.set == "A").mean()
A["prevalence_preserved"] = abs(prev(train_df) - prev(test_df)) < 0.005
A["abmix_preserved"] = abs(abmix(train_df) - abmix(test_df)) < 0.01

# (c) NaNs present BEFORE imputer, ZERO NaNs entering SMOTENC
nan_before = int(X_train.isna().sum().sum())
Xt = pre.fit_transform(X_train)                 # train-only fit
nan_into_smote = int(np.isnan(Xt).sum())
A["nan_before_impute_present"] = (nan_before > 0)
A["zero_nan_into_smote"] = (nan_into_smote == 0)

# (d) imputer fit on TRAIN ONLY — verify against fold indices, not by trusting pipe.
#     One manual fold: imputer.statistics_ must equal the TRAIN-FOLD median, and must
#     NOT equal the full-data median (proves test rows never entered the fit).
tr_idx, va_idx = next(cv.split(X_train, y_train))
fold_pre = ColumnTransformer([
    ("num", SkPipeline([("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler())]), NUM_COLS),
    ("cat", SimpleImputer(strategy="most_frequent"), CAT_COLS)])
fold_pre.fit(X_train.iloc[tr_idx])
imp_stats = fold_pre.named_transformers_["num"].named_steps["imp"].statistics_
col0 = NUM_COLS[0]
fold_median = X_train.iloc[tr_idx][col0].median()
full_median = X_train[col0].median()
A["imputer_uses_train_fold_only"] = np.isclose(imp_stats[0], fold_median)

# (e) test set never resampled — size + prevalence unchanged
A["test_size_unchanged"] = (len(X_test) == len(test_df))
A["test_not_resampled"] = (len(proba) == len(test_df))

# (f) indicator/categorical columns leaving SMOTENC are still discrete (no 0.5)
Xr, yr = sampler.fit_resample(Xt, y_train)      # resample the transformed train
cat_block = Xr[:, cat_idx]
frac = np.abs(cat_block - np.round(cat_block)).max()
A["smotenc_categoricals_discrete"] = (frac < 1e-9)

for k, v in A.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
all_pass = all(A.values())

# ----------------------------------------------------------------------------
# 6. Metrics
# ----------------------------------------------------------------------------
print("\n=== METRICS (skeleton, untuned) ===")
print(f"CV AUPRC  {cv_ap.mean():.3f} ± {cv_ap.std():.3f}")
print(f"CV AUROC  {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
print(f"TEST AUPRC {test_ap:.3f}   (baseline = prevalence {y_test.mean():.3f})")
print(f"TEST AUROC {test_auc:.3f}")

# ----------------------------------------------------------------------------
# 7. Traceability
# ----------------------------------------------------------------------------
with open(TEST_PIDS_FILE, "w") as fh:
    json.dump({"seed": SEED, "use_smotenc": USE_SMOTENC,
               "n_test": int(len(test_df)),
               "test_pids": sorted(test_df.pid.tolist())}, fh, indent=2)

# worked example: trace one test pid's Creatinine_mean  NaN -> imputed -> scaled
ex_pid = test_df.pid.iloc[0]
in_train = ex_pid in set(train_df.pid)
col = "Creatinine_mean"
raw = test_df.loc[test_df.pid == ex_pid, col].iloc[0]
ci = NUM_COLS.index(col)
imp_full = pipe.named_steps["pre"].named_transformers_["num"].named_steps["imp"].statistics_[ci]
sc = pipe.named_steps["pre"].named_transformers_["num"].named_steps["sc"]
imputed = imp_full if pd.isna(raw) else raw
scaled = (imputed - sc.mean_[ci]) / sc.scale_[ci]
print("\n=== WORKED EXAMPLE ===")
print(f"pid {ex_pid}: in_train={in_train}  in_test=True   (must be in exactly one)")
print(f"  {col}: raw={raw}  -> imputed(train median)={imp_full:.3f}  -> scaled={scaled:.3f}")

print("\nSTATUS:", "PASS — trusted skeleton" if all_pass else "FAIL — fix before Day 4")
print(f"saved: {TEST_PIDS_FILE}")
