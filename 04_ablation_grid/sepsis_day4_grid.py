"""
Day 4 · resampling x ablation GRID.

8 resampling conditions (ordered baseline->weight->under->over->hybrid)
x 3 feature arms (full, lab_core-ablated, all_7-ablated).
Headline = full vs lab_core (16 cells). all_7 = robustness companion (+8).

Guardrails: same split/folds/seed across every cell (reuses frozen Day-3 split);
class_weight is a sampler-free branch; cat_idx recomputed per arm; ablation drops
values AND _measured/_n; each cell written to disk on completion (resume safe).

Primary metric: drop-column dAUPRC = AUPRC(full) - AUPRC(ablated), per condition.
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
RESULTS_FILE, TEST_PIDS_FILE = "day4_grid_results.csv", "day3_test_pids.json"

ABLATION_ARMS = {
    "full": set(),
    "lab_core": {"Creatinine", "Platelets", "Bilirubin_total"},
    "all_7": {"Creatinine", "Platelets", "Bilirubin_total", "MAP", "FiO2", "SaO2", "O2Sat"},
}
BINARY_STATIC = ["Gender", "Unit1", "Unit2"]
CONDITIONS = [
    ("1_none",            "baseline", lambda ci: None,                                            None),
    ("2_class_weight",    "weight",   lambda ci: None,                                            "balanced"),
    ("3_RandomUnder",     "under",    lambda ci: RandomUnderSampler(random_state=SEED),           None),
    ("4_RandomOver",      "over",     lambda ci: RandomOverSampler(random_state=SEED),            None),
    ("5_SMOTENC",         "over",     lambda ci: SMOTENC(categorical_features=ci, random_state=SEED), None),
    ("6_BorderlineSMOTE", "over",     lambda ci: BorderlineSMOTE(random_state=SEED),              None),
    ("7_ADASYN",          "over",     lambda ci: ADASYN(random_state=SEED),                       None),
    ("8_SMOTETomek",      "hybrid",   lambda ci: SMOTETomek(random_state=SEED),                   None),
]

df = pd.read_parquet(FEATURES_FILE); tags = pd.read_csv(TAGS_FILE)
src_of = dict(zip(tags.feature, tags.source_var)); kind_of = dict(zip(tags.feature, tags.kind))
ID = ["pid", "set", "label"]; feature_cols = [c for c in df.columns if c not in ID]

def is_cat(c): return c in BINARY_STATIC or kind_of.get(c) == "indicator"
def arm_columns(ablated):
    keep = [c for c in feature_cols if src_of.get(c, c) not in ablated]
    num = [c for c in keep if not is_cat(c)]; cat = [c for c in keep if is_cat(c)]
    return num, cat, list(range(len(num), len(num) + len(cat)))
def make_pre(num, cat):
    return ColumnTransformer([
        ("num", SkPipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", SimpleImputer(strategy="most_frequent"), cat)])

df["strata"] = df["label"].astype(str) + "_" + df["set"].astype(str)
train_df, test_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df["strata"])
y_train, y_test = train_df["label"].to_numpy(), test_df["label"].to_numpy()
assert set(test_df.pid) == set(json.load(open(TEST_PIDS_FILE))["test_pids"]), "split drifted"
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

done = set()
if os.path.exists(RESULTS_FILE):
    done = set(zip(*[pd.read_csv(RESULTS_FILE)[c] for c in ("arm", "condition")]))

for arm, ablated in ABLATION_ARMS.items():
    num, cat, cat_idx = arm_columns(ablated)
    Xtr, Xte = train_df[num + cat], test_df[num + cat]
    for name, family, make_s, cw in CONDITIONS:
        if (arm, name) in done: continue
        t0 = time.time()
        try:
            pre = make_pre(num, cat)
            rf = RandomForestClassifier(n_estimators=N_TREES, class_weight=cw, random_state=SEED, n_jobs=1)
            s = make_s(cat_idx)
            steps = [("pre", pre)] + ([("smp", s)] if s is not None else []) + [("rf", rf)]
            pipe = ImbPipeline(steps)
            cv = cross_validate(pipe, Xtr, y_train, cv=skf, scoring=["average_precision", "roc_auc"], n_jobs=1)
            pipe.fit(Xtr, y_train); p = pipe.predict_proba(Xte)[:, 1]
            row = dict(arm=arm, condition=name, family=family,
                       cv_auprc=cv["test_average_precision"].mean(), cv_auprc_sd=cv["test_average_precision"].std(),
                       cv_auroc=cv["test_roc_auc"].mean(), cv_auroc_sd=cv["test_roc_auc"].std(),
                       test_auprc=average_precision_score(y_test, p), test_auroc=roc_auc_score(y_test, p),
                       secs=round(time.time() - t0, 1), error="")
        except Exception as e:
            row = dict(arm=arm, condition=name, family=family, cv_auprc=np.nan, cv_auprc_sd=np.nan,
                       cv_auroc=np.nan, cv_auroc_sd=np.nan, test_auprc=np.nan, test_auroc=np.nan,
                       secs=round(time.time() - t0, 1), error=str(e)[:120])
        hdr = not os.path.exists(RESULTS_FILE)
        pd.DataFrame([row]).to_csv(RESULTS_FILE, mode="a", index=False, header=hdr)
        v = f"{row['test_auprc']:.3f}" if pd.notna(row['test_auprc']) else "ERR"
        print(f"[{arm:8s} {name:16s}] test_auprc={v}  ({row['secs']}s)", flush=True)
print("\nGRID COMPLETE")
