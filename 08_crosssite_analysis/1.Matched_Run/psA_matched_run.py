"""
PS-A (matched) - cross-site transportability of measurement-ordering reliance.
Runs on the EXISTING matched features.parquet (both sites at 7.14% prevalence),
so this isolates covariate / measurement-practice shift with prevalence held fixed.

Two tiers, one resume-safe run:
  TIER 1  cross-site : train each arm ONCE on source site, score the target site.
                       Saves per-patient target predictions -> paired bootstrap later.
  TIER 2  within-src : 5x5 repeated CV within each source site (the in-distribution
                       reference), so the analyzer can form the difference-in-differences.

Outputs (appended, resume-safe) to OUT_DIR:
  psA_crosssite_preds.csv   long: direction, model, condition, arm, pid, y_true, prob
  psA_withinsrc_perfold.csv long: site, model, condition, arm, split, auprc, auroc

Free-tier sizing: default grid is 2 arms x 2 conditions x 3 models.
  Tier 1  ~24 fits (minutes).  Tier 2  ~600 fits (~1 h).  Resume-safe: rerun until DONE.

--- COLAB SETUP (run these in a cell FIRST) ---
from google.colab import drive; drive.mount('/content/drive')
!pip -q install imbalanced-learn xgboost lightgbm pyarrow
# put features.parquet + feature_tags.csv in DATA_DIR, point OUT_DIR at Drive.
"""
import os, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---------------- CONFIG ----------------
SEED, N_TREES = 42, 150
N_SPLITS, N_REPEATS = 5, 5                      # within-source 25 estimates
DATA_DIR = "."                                  # <- Drive path with the 2 data files
OUT_DIR  = "."                                  # <- Drive path for outputs
FEATURES_FILE = os.path.join(DATA_DIR, "features.parquet")
TAGS_FILE     = os.path.join(DATA_DIR, "feature_tags.csv")
CROSS_OUT   = os.path.join(OUT_DIR, "psA_crosssite_preds.csv")
WITHIN_OUT  = os.path.join(OUT_DIR, "psA_withinsrc_perfold.csv")

ARMS_TO_RUN  = ["full", "lab_indicators_only"]      # add "lab_panel" for the full decomposition
CONDITIONS_TO_RUN = ["1_none", "5_SMOTENC"]         # extend to all 8 if compute allows
MODELS = ["RF", "xgboost", "lightgbm"]
DIRECTIONS = [("A", "B"), ("B", "A")]              # (source, target)
RUN_TIER1_CROSS  = True
RUN_TIER2_WITHIN = True
# ----------------------------------------

BINARY_STATIC = ["Gender", "Unit1", "Unit2"]
SOFA_ALL7 = {"Creatinine","Platelets","Bilirubin_total","MAP","FiO2","SaO2","O2Sat"}

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
    # keep_empty_features=True: retain columns that are all-NaN in a single training site
    # (e.g. EtCO2 is never measured in site A). Without it the ColumnTransformer drops those
    # columns, the transformed width shrinks, and SMOTENC's categorical indices misalign -> IndexError.
    return ColumnTransformer([
        ("num", SkPipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                            ("sc", StandardScaler())]), num),
        ("cat", SimpleImputer(strategy="most_frequent", keep_empty_features=True), cat)])

def make_model(name, pos_weight, use_weight):
    if name == "RF":
        return RandomForestClassifier(n_estimators=N_TREES, random_state=SEED, n_jobs=1,
                                      class_weight=("balanced" if use_weight else None))
    if name == "xgboost":
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
                             colsample_bytree=0.9, tree_method="hist", eval_metric="logloss",
                             random_state=SEED, n_jobs=1,
                             scale_pos_weight=(pos_weight if use_weight else 1))
    if name == "lightgbm":
        return LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.1, subsample=0.9,
                              colsample_bytree=0.9, random_state=SEED, n_jobs=1, verbose=-1,
                              class_weight=("balanced" if use_weight else None))
    raise ValueError(name)

def build_pipe(arm, model, condition, num, cat, cat_idx, y):
    """condition '1_none' = no resampling; '5_SMOTENC' = SMOTENC oversample train."""
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    pre = make_pre(num, cat)
    est = make_model(model, pos_weight, use_weight=False)
    steps = [("pre", pre)]
    if condition == "5_SMOTENC":
        steps.append(("smp", SMOTENC(categorical_features=cat_idx, random_state=SEED)))
    steps.append(("clf", est))
    return ImbPipeline(steps)

# ---------------- TIER 1 : cross-site ----------------
if RUN_TIER1_CROSS:
    done = set()
    if os.path.exists(CROSS_OUT):
        d = pd.read_csv(CROSS_OUT, usecols=["direction","model","condition","arm"])
        done = set(map(tuple, d.drop_duplicates().values))
    for src, tgt in DIRECTIONS:
        s_df = df[df.set == src]; t_df = df[df.set == tgt]
        ys = s_df.label.to_numpy(); yt = t_df.label.to_numpy()
        for model in MODELS:
            for cond in CONDITIONS_TO_RUN:
                for arm in ARMS_TO_RUN:
                    key = (f"{src}->{tgt}", model, cond, arm)
                    if key in done:
                        print("skip", key); continue
                    t0 = time.time()
                    num, cat, cat_idx = arm_columns(arm)
                    pipe = build_pipe(arm, model, cond, num, cat, cat_idx, ys)
                    pipe.fit(s_df[num + cat], ys)
                    prob = pipe.predict_proba(t_df[num + cat])[:, 1]
                    rows = pd.DataFrame({"direction": f"{src}->{tgt}", "model": model,
                                         "condition": cond, "arm": arm,
                                         "pid": t_df.pid.values, "y_true": yt, "prob": prob})
                    rows.to_csv(CROSS_OUT, mode="a", index=False, header=not os.path.exists(CROSS_OUT))
                    ap = average_precision_score(yt, prob)
                    print(f"[X {src}->{tgt} {model:9s} {cond:9s} {arm:20s}] AUPRC={ap:.3f} ({time.time()-t0:.0f}s)", flush=True)
    print("TIER 1 CROSS-SITE DONE")

# ---------------- TIER 2 : within-source CV ----------------
if RUN_TIER2_WITHIN:
    done = set()
    if os.path.exists(WITHIN_OUT):
        d = pd.read_csv(WITHIN_OUT, usecols=["site","model","condition","arm"])
        done = set(map(tuple, d.drop_duplicates().values))
    rcv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    for site in sorted(df.set.unique()):
        s_df = df[df.set == site]; ys = s_df.label.to_numpy()
        for model in MODELS:
            for cond in CONDITIONS_TO_RUN:
                for arm in ARMS_TO_RUN:
                    key = (site, model, cond, arm)
                    if key in done:
                        print("skip", key); continue
                    t0 = time.time()
                    num, cat, cat_idx = arm_columns(arm)
                    pipe = build_pipe(arm, model, cond, num, cat, cat_idx, ys)
                    cv = cross_validate(pipe, s_df[num + cat], ys, cv=rcv,
                                        scoring=["average_precision", "roc_auc"], n_jobs=1)
                    rows = [dict(site=site, model=model, condition=cond, arm=arm, split=i,
                                 auprc=cv["test_average_precision"][i], auroc=cv["test_roc_auc"][i])
                            for i in range(N_SPLITS * N_REPEATS)]
                    pd.DataFrame(rows).to_csv(WITHIN_OUT, mode="a", index=False, header=not os.path.exists(WITHIN_OUT))
                    print(f"[W {site} {model:9s} {cond:9s} {arm:20s}] AUPRC={np.mean([r['auprc'] for r in rows]):.3f} ({time.time()-t0:.0f}s)", flush=True)
    print("TIER 2 WITHIN-SOURCE DONE")
