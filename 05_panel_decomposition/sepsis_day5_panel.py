"""
Day 5 · whole-lab-panel ablation  (Option B).

Closes the Day-4 gap. Day-4 found the 3 lab-core SOFA indicators were collinear
with NON-SOFA lab indicators (Creatinine<->BUN 0.97, etc.), so component-wise
ablation couldn't isolate lab circularity. The fix is panel-level ablation.

Two new arms (same harness, frozen split, RF, 8 conditions):
  lab_panel            : drop ALL 26 labs (values + _measured/_n indicators)
                         -> full - lab_panel = TOTAL lab contribution
  lab_indicators_only  : drop ALL lab _measured/_n indicators, KEEP lab values
                         -> full - lab_indicators_only = ordering signal beyond
                            value = CIRCULARITY, isolated  (the thesis number)

To run only the minimal spec arm, set ARMS_TO_RUN = ["lab_panel"].

Reuses the existing results files so cells stay comparable to Day 4:
  appends summary rows to day4_grid_results.csv   (for the gradient)
  appends per-fold rows to day4_perfold.csv       (for the paired test, Option C)
Resume-safe. Single-core ~35-40 min for both arms (panel arm is fast; indicators
arm keeps values so it's slower). Set RF n_jobs=-1 if you have cores.
"""
import os, json, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC, RandomOverSampler, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

SEED, N_TREES = 42, 150
FEATURES_FILE, TAGS_FILE = "features.parquet", "feature_tags.csv"
TEST_PIDS_FILE = "day3_test_pids.json"
GRID_FILE, PERFOLD_FILE = "day4_grid_results.csv", "day4_perfold.csv"

ARMS_TO_RUN = ["lab_panel", "lab_indicators_only"]   # set to ["lab_panel"] for minimal spec
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
LAB_VARS = set(tags[tags.group == "lab"].source_var.unique())   # all 26 labs

def is_cat(c): return c in BINARY_STATIC or kind_of.get(c) == "indicator"

# drop predicate per arm
DROP = {
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
y_train, y_test = train_df.label.to_numpy(), test_df.label.to_numpy()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

done_grid = set()
if os.path.exists(GRID_FILE):
    g = pd.read_csv(GRID_FILE); done_grid = set(zip(g.arm, g.condition))

for arm in ARMS_TO_RUN:
    num, cat, cat_idx = arm_columns(arm)
    print(f"\narm {arm}: {len(num)+len(cat)} features ({len(num)} num, {len(cat)} cat)")
    Xtr, Xte = train_df[num + cat], test_df[num + cat]
    for name, make_s, cw in CONDITIONS:
        if (arm, name) in done_grid:
            print(f"  skip {name} (already done)"); continue
        t0 = time.time()
        pre = make_pre(num, cat)
        rf = RandomForestClassifier(n_estimators=N_TREES, class_weight=cw, random_state=SEED, n_jobs=1)
        s = make_s(cat_idx)
        steps = [("pre", pre)] + ([("smp", s)] if s is not None else []) + [("rf", rf)]
        pipe = ImbPipeline(steps)
        cv = cross_validate(pipe, Xtr, y_train, cv=skf,
                            scoring=["average_precision", "roc_auc"], n_jobs=1)
        pipe.fit(Xtr, y_train); p = pipe.predict_proba(Xte)[:, 1]
        # summary row -> grid file
        summ = dict(arm=arm, condition=name, family="ablation",
                    cv_auprc=cv["test_average_precision"].mean(), cv_auprc_sd=cv["test_average_precision"].std(),
                    cv_auroc=cv["test_roc_auc"].mean(), cv_auroc_sd=cv["test_roc_auc"].std(),
                    test_auprc=average_precision_score(y_test, p), test_auroc=roc_auc_score(y_test, p),
                    secs=round(time.time() - t0, 1), error="")
        pd.DataFrame([summ]).to_csv(GRID_FILE, mode="a", index=False, header=not os.path.exists(GRID_FILE))
        # per-fold rows -> perfold file
        pf = [dict(arm=arm, condition=name, fold=i,
                   auprc=cv["test_average_precision"][i], auroc=cv["test_roc_auc"][i]) for i in range(5)]
        pd.DataFrame(pf).to_csv(PERFOLD_FILE, mode="a", index=False, header=not os.path.exists(PERFOLD_FILE))
        print(f"  [{name:16s}] test_auprc={summ['test_auprc']:.3f}  ({summ['secs']}s)", flush=True)

print("\nDAY 5 ARMS COMPLETE")
