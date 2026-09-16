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

--- COLAB SETUP ---
Upload to /content:  the script, the features parquet, feature_tags.csv,
                     and BOTH existing output CSVs (so resume fires).
!pip -q install imbalanced-learn xgboost lightgbm pyarrow
%run psA_matched_run_native.py          <- use %run, NOT !python (drive.mount needs the kernel)
Outputs are written to Drive (DRIVE_OUT_DIR) and survive runtime disconnects.
On first run the uploaded output CSVs are copied from /content to Drive automatically.
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

DATA_DIR = "/content"                           # where you uploaded the inputs

# /content is EPHEMERAL. It is wiped on every runtime reset, and Tier 2 is a ~1h job
# that appends as it goes. If the runtime drops with outputs in /content you lose the
# new cells AND the prior outputs resume depends on -> full restart from zero.
# Writing outputs to Drive makes a disconnect cost one cell instead of an hour.
USE_DRIVE_FOR_OUTPUTS = True                    # False -> keep outputs in DATA_DIR
DRIVE_OUT_DIR = "/content/drive/MyDrive/psa_out"

def _resolve_out_dir():
    if not USE_DRIVE_FOR_OUTPUTS:
        return DATA_DIR
    # Already mounted from a notebook cell? Use it. The mount is a filesystem-level
    # FUSE mount, so a subprocess launched via `!python script.py` can see it.
    if os.path.exists("/content/drive/MyDrive"):
        os.makedirs(DRIVE_OUT_DIR, exist_ok=True)
        return DRIVE_OUT_DIR
    # Not mounted. drive.mount() needs the IPython kernel to raise the auth prompt,
    # so it CANNOT run from a subprocess -> "'NoneType' object has no attribute 'kernel'".
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        os.makedirs(DRIVE_OUT_DIR, exist_ok=True)
        return DRIVE_OUT_DIR
    except Exception as e:
        print("[warn] " + "-" * 70)
        print(f"[warn] Drive not mounted ({type(e).__name__}). Outputs -> {DATA_DIR} (EPHEMERAL).")
        print("[warn]")
        print("[warn] drive.mount() needs the IPython kernel, so it cannot run from a")
        print("[warn] subprocess. `!python script.py` is a subprocess. Two ways to fix:")
        print("[warn]")
        print("[warn]   (a) launch in-kernel instead -- the mount below will then work:")
        print("[warn]           %run psA_matched_run_native.py")
        print("[warn]")
        print("[warn]   (b) or mount once from a notebook cell, then !python is fine:")
        print("[warn]           from google.colab import drive")
        print("[warn]           drive.mount('/content/drive')")
        print("[warn] " + "-" * 70)
        return DATA_DIR

OUT_DIR = _resolve_out_dir()

FEATURES_FILE = os.path.join(DATA_DIR, "features_native.parquet")
TAGS_FILE     = os.path.join(DATA_DIR, "feature_tags.csv")          # original, not _native
CROSS_OUT   = os.path.join(OUT_DIR, "psA_native_crosssite_preds.csv")
WITHIN_OUT  = os.path.join(OUT_DIR, "psA_native_withinsrc_perfold.csv")

ARMS_TO_RUN = ["full", "lab_indicators_only", "ordering_only", "values_only", "lab_panel"]
CONDITIONS_TO_RUN = ["1_none", "5_SMOTENC"]        # SMOTENC x standalone arms are skipped below
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
    "ordering_only": lambda c: not (src_of.get(c, c) in LAB_VARS
                                and kind_of.get(c) == "indicator"),
    "values_only":   lambda c: not (src_of.get(c, c) in LAB_VARS
                                and kind_of.get(c) == "value"),
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

# ---------------- PREFLIGHT (added: resume verification + duplicate repair) ----------------
# The failure mode this prevents: if the existing output CSVs are not in OUT_DIR,
# os.path.exists() returns False, `done` stays empty, every completed cell is refit,
# and the results are APPENDED as duplicate rows. The analyzer pools by
# (direction, model, condition, arm), so duplicates silently double the bootstrap
# sample and shrink the CIs. Corrupted output that looks completely normal.

EXPECT_EXISTING_OUTPUTS = True   # set False ONLY for a genuine from-scratch run
REPAIR_DUPLICATES       = True   # de-dupe existing outputs in place (writes .bak first)

CROSS_KEY  = ["direction", "model", "condition", "arm", "pid"]
WITHIN_KEY = ["site", "model", "condition", "arm", "split"]

SEARCH_FOR_PRIOR_OUTPUTS = True   # hunt for prior output CSVs anywhere under Drive//content

def _find_prior_output(name):
    """Locate a prior output CSV. Checks the obvious paths, then recursively searches
    Drive and /content. Returns the largest match (most completed cells) or None."""
    for c in (os.path.join(OUT_DIR, name), os.path.join(DATA_DIR, name),
              os.path.join("/content", name), name):
        if os.path.exists(c):
            return os.path.abspath(c)
    if not SEARCH_FOR_PRIOR_OUTPUTS:
        return None
    import glob
    hits = []
    for root in ("/content/drive/MyDrive", "/content"):
        if os.path.isdir(root):
            hits += glob.glob(os.path.join(root, "**", name), recursive=True)
    if not hits:
        return None
    return sorted(set(hits), key=lambda p: -os.path.getsize(p))[0]


def _seed_outputs():
    """Copy any prior outputs we can find into OUT_DIR so resume fires."""
    import shutil
    for path in (CROSS_OUT, WITHIN_OUT):
        name = os.path.basename(path)
        if os.path.exists(path):
            continue
        found = _find_prior_output(name)
        if found and os.path.abspath(found) != os.path.abspath(path):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            shutil.copy2(found, path)
            print(f"  found prior {name}")
            print(f"    at   {found}")
            print(f"    -> copied into OUT_DIR")

def _preflight():
    print("=" * 78)
    print("PREFLIGHT")
    print("=" * 78)
    print(f"  DATA_DIR -> {os.path.abspath(DATA_DIR)}")
    print(f"  OUT_DIR  -> {os.path.abspath(OUT_DIR)}")

    for f in (FEATURES_FILE, TAGS_FILE):
        if not os.path.exists(f):
            raise SystemExit(f"\nABORT: input file not found: {os.path.abspath(f)}")
        print(f"  input OK: {os.path.basename(f)}")

    print(f"  features: n={len(df)}  prevalence={df.label.mean():.4f}  "
          f"sites={dict(df.groupby('set').size())}")

    print("\n  arm widths (expect 52|0/52, 130|130/0, 61|42/19):")
    for a in ARMS_TO_RUN:
        n, c, _ = arm_columns(a)
        print(f"    {a:22s} total={len(n)+len(c):3d} | num={len(n):3d} cat={len(c):3d}")

    _seed_outputs()
    print("\n  existing outputs:")
    missing = []
    for path, key in ((CROSS_OUT, CROSS_KEY), (WITHIN_OUT, WITHIN_KEY)):
        name = os.path.basename(path)
        if not os.path.exists(path):
            print(f"    {name:34s} NOT FOUND")
            missing.append(name)
            continue
        d = pd.read_csv(path)
        dup = int(d.duplicated(subset=key).sum())
        arms = sorted(d.arm.unique())
        print(f"    {name:34s} {len(d):>8,} rows | arms={arms} | duplicate rows={dup:,}")
        if dup:
            if not REPAIR_DUPLICATES:
                raise SystemExit(f"\nABORT: {dup:,} duplicate rows in {name}. "
                                 f"Set REPAIR_DUPLICATES=True or restore a clean copy.")
            bak = path + ".bak"
            if not os.path.exists(bak):
                d.to_csv(bak, index=False)
                print(f"      backup written -> {os.path.basename(bak)}")
            clean = d.drop_duplicates(subset=key, keep="first")
            clean.to_csv(path, index=False)
            print(f"      REPAIRED: {len(d):,} -> {len(clean):,} rows "
                  f"({dup:,} duplicates removed; values are deterministic, keep='first' is safe)")

    if missing and EXPECT_EXISTING_OUTPUTS:
        raise SystemExit(
            "\nABORT: expected prior outputs not found in OUT_DIR: " + ", ".join(missing) +
            "\n       Resume would not fire and completed cells would be refit and"
            "\n       appended as duplicates."
            "\n       Fix: copy the existing CSVs into OUT_DIR, or set"
            "\n       EXPECT_EXISTING_OUTPUTS=False if this really is a fresh run.")

    print("\n  PREFLIGHT PASSED -> expect a wall of 'skip (...)' lines next.")
    print("=" * 78 + "\n")

_preflight()

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
                    
                    if cond == "5_SMOTENC" and arm in {"ordering_only", "values_only"}:
                        continue
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
                    if cond == "5_SMOTENC" and arm in {"ordering_only", "values_only"}:
                        continue
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
