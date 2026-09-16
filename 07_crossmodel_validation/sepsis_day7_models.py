"""
Day 7 · cross-model generalization  (Option A).

Re-runs the Day-6 repeated-CV grid with XGBoost and LightGBM, to test whether the
Day-6 NEGATIVE result (no significant resampling-amplification of circularity)
holds across model families, not just Random Forest. For a null, robustness across
models is what makes it convincing.

LOCKED DESIGN — impute uniformly, do NOT use native NaN handling.
XGBoost and LightGBM can ingest NaN, but allowing that would change the missingness
regime per model and confound the comparison. The pipeline is IDENTICAL to RF:
  impute(median) -> scale -> sampler -> model
so the only thing changing across the three models is the model itself. Scaling is
harmless for trees; it stays for SMOTENC's distance metric, same as RF.

The 'class_weight' condition uses each model's idiomatic balanced weighting:
  RF (Day 6): class_weight='balanced'
  XGBoost   : scale_pos_weight = n_neg/n_pos
  LightGBM  : class_weight='balanced'
(documented asymmetry — each family weights differently; there is no shared knob.)

Same frozen split, same 25-fold RepeatedStratifiedKFold, same seed, same arms and
conditions as Day 6. RF is NOT re-run here; the analysis reads RF from
day6_repeated_perfold.csv.

*** COMPUTE ***  2 models x 4 arms x 8 conditions x 25 fits. LightGBM is fast;
XGBoost moderate -- likely comparable to or faster than Day 6's RF per fit, but
still multi-hour single-core. Resume-safe: rerun until "DAY 7 COMPLETE". Set
n_jobs=-1 in the model constructors if you have cores.

Output: day7_repeated_perfold.csv  (model, arm, condition, split, auprc, auroc)
"""
import os, json, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC, RandomOverSampler, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42
N_SPLITS, N_REPEATS = 5, 5
FEATURES_FILE, TAGS_FILE = "features.parquet", "feature_tags.csv"
TEST_PIDS_FILE, OUT = "day3_test_pids.json", "day7_repeated_perfold.csv"

MODELS = ["xgboost", "lightgbm"]            # RF already in day6_repeated_perfold.csv
ARMS_TO_RUN = ["full", "lab_panel", "lab_indicators_only", "all_7"]
SOFA_ALL7 = {"Creatinine", "Platelets", "Bilirubin_total", "MAP", "FiO2", "SaO2", "O2Sat"}
BINARY_STATIC = ["Gender", "Unit1", "Unit2"]

# (name, sampler factory, uses_class_weight)
CONDITIONS = [
    ("1_none",            lambda ci: None,                                            False),
    ("2_class_weight",    lambda ci: None,                                            True),
    ("3_RandomUnder",     lambda ci: RandomUnderSampler(random_state=SEED),           False),
    ("4_RandomOver",      lambda ci: RandomOverSampler(random_state=SEED),            False),
    ("5_SMOTENC",         lambda ci: SMOTENC(categorical_features=ci, random_state=SEED), False),
    ("6_BorderlineSMOTE", lambda ci: BorderlineSMOTE(random_state=SEED),              False),
    ("7_ADASYN",          lambda ci: ADASYN(random_state=SEED),                       False),
    ("8_SMOTETomek",      lambda ci: SMOTETomek(random_state=SEED),                   False),
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

def make_model(name, use_weight, pos_weight):
    if name == "xgboost":
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                             eval_metric="logloss", random_state=SEED, n_jobs=1,
                             scale_pos_weight=(pos_weight if use_weight else 1))
    if name == "lightgbm":
        return LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.1,
                              subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                              n_jobs=1, verbose=-1,
                              class_weight=("balanced" if use_weight else None))
    raise ValueError(name)

df["strata"] = df["label"].astype(str) + "_" + df["set"].astype(str)
train_df, test_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df["strata"])
assert set(test_df.pid) == set(json.load(open(TEST_PIDS_FILE))["test_pids"]), "split drifted"
y_train = train_df.label.to_numpy()
pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)   # n_neg/n_pos
rcv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

done = set()
if os.path.exists(OUT):
    p = pd.read_csv(OUT); done = set(zip(p.model, p.arm, p.condition))

for model_name in MODELS:
    for arm in ARMS_TO_RUN:
        num, cat, cat_idx = arm_columns(arm)
        Xtr = train_df[num + cat]
        print(f"\n[{model_name}] arm {arm}: {len(num)+len(cat)} features", flush=True)
        for name, make_s, use_w in CONDITIONS:
            if (model_name, arm, name) in done:
                print(f"  skip {name}"); continue
            t0 = time.time()
            pre = make_pre(num, cat)
            mdl = make_model(model_name, use_w, pos_weight)
            s = make_s(cat_idx)
            steps = [("pre", pre)] + ([("smp", s)] if s is not None else []) + [("mdl", mdl)]
            cv = cross_validate(ImbPipeline(steps), Xtr, y_train, cv=rcv,
                                scoring=["average_precision", "roc_auc"], n_jobs=1)
            rows = [dict(model=model_name, arm=arm, condition=name, split=i,
                         auprc=cv["test_average_precision"][i], auroc=cv["test_roc_auc"][i])
                    for i in range(N_SPLITS * N_REPEATS)]
            pd.DataFrame(rows).to_csv(OUT, mode="a", index=False, header=not os.path.exists(OUT))
            print(f"  [{name:16s}] saved  ({time.time()-t0:.0f}s)", flush=True)

print("\nDAY 7 COMPLETE")
