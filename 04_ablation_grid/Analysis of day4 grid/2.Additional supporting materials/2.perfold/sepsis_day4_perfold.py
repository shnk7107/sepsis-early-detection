"""
Day 4 · per-fold capture  (RUN SECOND, only if the redundancy check is clean).

The original grid saved CV mean/sd but discarded the 5 per-fold scores, so a paired
test (which needs the matched per-fold values) is impossible from the results CSV.
This re-runs the SAME CV -- same split, folds, seed, 150 trees -- and saves every
fold's AUPRC and AUROC. It SKIPS the final test fit (test scores already exist in
day4_grid_results.csv), so it's ~17% cheaper than the original grid.

Folds are identical across arms (StratifiedKFold depends on y + seed, not on which
feature columns are present), so fold i of `full` and fold i of `lab_core` contain
the SAME patients -> the per-fold deltas are properly paired.

Resume-safe: each cell appends to day4_perfold.csv. Re-run until GRID COMPLETE.
Single-core: ~30-35 min. Cut it short after `full` + `lab_core` (16 cells) if you
only need the headline paired test; `all_7` is the companion.

Output: day4_perfold.csv   (long format: arm, condition, fold, auprc, auroc)
"""
import os, json, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
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
FEATURES_FILE, TAGS_FILE = "features.parquet", "feature_tags.csv"
TEST_PIDS_FILE, OUT = "day3_test_pids.json", "day4_perfold.csv"

ABLATION_ARMS = {
    "full": set(),
    "lab_core": {"Creatinine", "Platelets", "Bilirubin_total"},
    "all_7": {"Creatinine", "Platelets", "Bilirubin_total", "MAP", "FiO2", "SaO2", "O2Sat"},
}
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
def is_cat(c): return c in BINARY_STATIC or kind_of.get(c) == "indicator"
def arm_columns(ab):
    keep = [c for c in feature_cols if src_of.get(c, c) not in ab]
    num = [c for c in keep if not is_cat(c)]; cat = [c for c in keep if is_cat(c)]
    return num, cat, list(range(len(num), len(num) + len(cat)))
def make_pre(num, cat):
    return ColumnTransformer([
        ("num", SkPipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", SimpleImputer(strategy="most_frequent"), cat)])

df["strata"] = df["label"].astype(str) + "_" + df["set"].astype(str)
train_df, _ = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df["strata"])
# (test set unused here; we only re-run CV. Assert split parity for safety.)
_, test_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df["strata"])
assert set(test_df.pid) == set(json.load(open(TEST_PIDS_FILE))["test_pids"]), "split drifted"
y_train = train_df.label.to_numpy()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

done = set()
if os.path.exists(OUT):
    p = pd.read_csv(OUT); done = set(zip(p.arm, p.condition))

for arm, ab in ABLATION_ARMS.items():
    num, cat, cat_idx = arm_columns(ab)
    Xtr = train_df[num + cat]
    for name, make_s, cw in CONDITIONS:
        if (arm, name) in done: continue
        t0 = time.time()
        pre = make_pre(num, cat)
        rf = RandomForestClassifier(n_estimators=N_TREES, class_weight=cw, random_state=SEED, n_jobs=1)
        s = make_s(cat_idx)
        steps = [("pre", pre)] + ([("smp", s)] if s is not None else []) + [("rf", rf)]
        cv = cross_validate(ImbPipeline(steps), Xtr, y_train, cv=skf,
                            scoring=["average_precision", "roc_auc"], n_jobs=1)
        rows = [dict(arm=arm, condition=name, fold=i,
                     auprc=cv["test_average_precision"][i], auroc=cv["test_roc_auc"][i])
                for i in range(5)]
        hdr = not os.path.exists(OUT)
        pd.DataFrame(rows).to_csv(OUT, mode="a", index=False, header=hdr)
        print(f"[{arm:8s} {name:16s}] 5 folds saved  ({time.time()-t0:.0f}s)", flush=True)
print("\nGRID COMPLETE")
