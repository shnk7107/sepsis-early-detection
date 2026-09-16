"""
Day 6 · repeated CV for significance  (Option C).

Replaces the single 5-fold CV with RepeatedStratifiedKFold (5 folds x 5 repeats =
25 estimates per cell), so the paired tests run on 25 matched deltas instead of 5.
This is what turns the Day-5 "suggestive but underpowered" gradients into a
defensible significance claim -- or shows they were fold noise.

SCOPE: only the arms with open, underpowered claims.
  full, all_7, lab_indicators_only, lab_panel.
  lab_core is EXCLUDED -- it is a null; repeated CV cannot power a null into
  existence, so running it would only waste compute.

Repeated-CV splits depend on (y, seed), not on which feature columns are present,
so split i is the SAME patient partition across arms -> the 25 deltas are paired.

*** COMPUTE WARNING ***  This is the expensive day: 25 fits per cell, ~5x the grid.
4 arms x 8 conditions = 32 cells. Single-core Colab = several hours. Resume-safe:
rerun until "DAY 6 COMPLETE". If you have cores, set RF n_jobs=-1 (big speedup).
To cut scope, drop "lab_panel" from ARMS_TO_RUN (keeps the two key gradients).

Output: day6_repeated_perfold.csv  (long: arm, condition, split, auprc, auroc)
"""
import os, json, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC, RandomOverSampler, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

SEED, N_TREES = 42, 150
N_SPLITS, N_REPEATS = 5, 5                      # 25 estimates / cell
FEATURES_FILE, TAGS_FILE = "features.parquet", "feature_tags.csv"
TEST_PIDS_FILE, OUT = "day3_test_pids.json", "day6_repeated_perfold.csv"

ARMS_TO_RUN = ["full", "lab_panel", "lab_indicators_only", "all_7"]   # drop lab_panel to cut compute
SOFA_ALL7 = {"Creatinine", "Platelets", "Bilirubin_total", "MAP", "FiO2", "SaO2", "O2Sat"}
BINARY_STATIC = ["Gender", "Unit1", "Unit2"]
CONDITIONS = [
    ("1_none",            lambda ci: None,                                            None),
    ("2_class_weight",    lambda ci: None,                                            "balanced"),
    ("3_RandomUnder",     lambda ci: RandomUnderSampler(random_state=SEED),           None),
    ("4_RandomOver",      lambda ci: RandomOverSampler(random_state=SEED),            None),
    ("5_SMOTENC",         lambda ci: SMOTENC(categorical_features=ci, random_state=SEED), None),
    ("6_BorderlineSMOTE", lambda ci: BorderlineSMOTE(random_state=SEED),              None),
    ("7_ADASYN",          lambda ci: ADASYN(random_state=SEED),                       None),
    ("8_SMOTETomek",      lambda ci: SMOTETomek(random_state=SEED),                   None),
]

df = pd.read_parquet(FEATURES_FILE); tags = pd.read_csv(TAGS_FILE)
src_of = dict(zip(tags.feature, tags.source_var)); kind_of = dict(zip(tags.feature, tags.kind))
ID = ["pid", "set", "label"]; feature_cols = [c for c in df.columns if c not in ID]
LAB_VARS = set(tags[tags.group == "lab"].source_var.unique())
def is_cat(c): return c in BINARY_STATIC or kind_of.get(c) == "indicator"

DROP = {
    "full":                lambda c: False,
    "all_7":               lambda c: src_of.get(c, c) in SOFA_ALL7,
    "lab_panel":           lambda c: src_of.get(c, c) in LAB_VARS,
    "lab_indicators_only": lambda c: src_of.get(c, c) in LAB_VARS and kind_of.get(c) == "indicator",
}
def arm_columns(arm):
    drop = DROP[arm]
    keep = [c for c in feature_cols if not drop(c)]
    num = [c for c in keep if not is_cat(c)]; cat = [c for c in keep if is_cat(c)]
    return num, cat, list(range(len(num), len(num) + len(cat)))
def make_pre(num, cat):
    return ColumnTransformer([
        ("num", SkPipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", SimpleImputer(strategy="most_frequent"), cat)])

df["strata"] = df["label"].astype(str) + "_" + df["set"].astype(str)
train_df, test_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df["strata"])
assert set(test_df.pid) == set(json.load(open(TEST_PIDS_FILE))["test_pids"]), "split drifted"
y_train = train_df.label.to_numpy()
rcv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

done = set()
if os.path.exists(OUT):
    p = pd.read_csv(OUT); done = set(zip(p.arm, p.condition))

for arm in ARMS_TO_RUN:
    num, cat, cat_idx = arm_columns(arm)
    print(f"\narm {arm}: {len(num)+len(cat)} features", flush=True)
    Xtr = train_df[num + cat]
    for name, make_s, cw in CONDITIONS:
        if (arm, name) in done:
            print(f"  skip {name}"); continue
        t0 = time.time()
        pre = make_pre(num, cat)
        rf = RandomForestClassifier(n_estimators=N_TREES, class_weight=cw, random_state=SEED, n_jobs=1)
        s = make_s(cat_idx)
        steps = [("pre", pre)] + ([("smp", s)] if s is not None else []) + [("rf", rf)]
        cv = cross_validate(ImbPipeline(steps), Xtr, y_train, cv=rcv,
                            scoring=["average_precision", "roc_auc"], n_jobs=1)
        rows = [dict(arm=arm, condition=name, split=i,
                     auprc=cv["test_average_precision"][i], auroc=cv["test_roc_auc"][i])
                for i in range(N_SPLITS * N_REPEATS)]
        pd.DataFrame(rows).to_csv(OUT, mode="a", index=False, header=not os.path.exists(OUT))
        print(f"  [{name:16s}] {N_SPLITS*N_REPEATS} splits saved  ({time.time()-t0:.0f}s)", flush=True)

print("\nDAY 6 COMPLETE")
