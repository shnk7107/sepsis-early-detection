"""
Day 4 · redundancy check  (RUN THIS FIRST).

Question it answers: is the lab-core null real, or an artifact of the SOFA
indicators being redundant with other features?

Drop-column dAUPRC ~= 0 can mean either:
  (a) the SOFA features are genuinely unimportant            -> null is REAL, or
  (b) they ARE used, but other features substitute when they're dropped
                                                             -> null is an ARTIFACT.

Permutation importance breaks the ambiguity: it permutes ONE feature while keeping
all others intact, so it measures marginal value GIVEN the rest.
  - lab-core permutation importance ~= 0  AND drop-column ~= 0  -> null REAL.
  - lab-core permutation importance >> 0  BUT drop-column ~= 0   -> redundancy ARTIFACT.

We fit two cells on the frozen split -- baseline (1_none) and most-aggressive
(8_SMOTETomek), both FULL arm -- and compare grouped permutation importance.
A correlation screen on the indicators is included as corroboration.

Outputs: day4_permimp.csv  + a console verdict.
Compute: ~10 min single-core (2 fits + permutation passes). Set n_jobs if you have cores.
"""
import json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek

SEED, N_TREES, N_REPEATS = 42, 150, 5
FEATURES_FILE, TAGS_FILE, TEST_PIDS_FILE = "features.parquet", "feature_tags.csv", "day3_test_pids.json"
LAB_CORE = {"Creatinine", "Platelets", "Bilirubin_total"}
ALL7 = LAB_CORE | {"MAP", "FiO2", "SaO2", "O2Sat"}
BINARY_STATIC = ["Gender", "Unit1", "Unit2"]

df = pd.read_parquet(FEATURES_FILE); tags = pd.read_csv(TAGS_FILE)
src_of = dict(zip(tags.feature, tags.source_var)); kind_of = dict(zip(tags.feature, tags.kind))
ID = ["pid", "set", "label"]; feats = [c for c in df.columns if c not in ID]
def is_cat(c): return c in BINARY_STATIC or kind_of.get(c) == "indicator"
num = [c for c in feats if not is_cat(c)]; cat = [c for c in feats if is_cat(c)]
cat_idx = list(range(len(num), len(num) + len(cat)))

df["strata"] = df["label"].astype(str) + "_" + df["set"].astype(str)
train_df, test_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df["strata"])
assert set(test_df.pid) == set(json.load(open(TEST_PIDS_FILE))["test_pids"]), "split drifted"
Xtr, Xte = train_df[num + cat], test_df[num + cat]
ytr, yte = train_df.label.to_numpy(), test_df.label.to_numpy()

def make_pipe(sampler):
    pre = ColumnTransformer([
        ("num", SkPipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", SimpleImputer(strategy="most_frequent"), cat)])
    rf = RandomForestClassifier(n_estimators=N_TREES, random_state=SEED, n_jobs=1)
    steps = [("pre", pre)] + ([("smp", sampler)] if sampler else []) + [("rf", rf)]
    return ImbPipeline(steps)

def group_of(col):
    s = src_of.get(col, col)
    if s in LAB_CORE: return "SOFA_lab_core"
    if s in ALL7:     return "SOFA_other(MAP+resp)"
    return "non_SOFA"

cells = {"1_none_baseline": None, "8_SMOTETomek_aggressive": SMOTETomek(random_state=SEED)}
records = []
order = num + cat
for label, sampler in cells.items():
    print(f"fitting {label} ...", flush=True)
    pipe = make_pipe(sampler); pipe.fit(Xtr, ytr)
    pi = permutation_importance(pipe, Xte, yte, scoring="average_precision",
                                n_repeats=N_REPEATS, random_state=SEED, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=order)
    grp = pd.Series({c: group_of(c) for c in order})
    summary = imp.groupby(grp).sum()
    print(f"  summed permutation importance (AUPRC drop) by group:")
    for g, v in summary.items():
        print(f"    {g:24s} {v:+.4f}")
    for c in order:
        records.append(dict(cell=label, feature=c, group=group_of(c),
                            perm_importance=imp[c]))

pd.DataFrame(records).to_csv("day4_permimp.csv", index=False)

# ---- correlation redundancy screen on lab-core SOFA indicators ----
print("\n=== correlation screen: lab-core SOFA indicators vs all other indicators ===")
ind_cols = [c for c in feats if kind_of.get(c) == "indicator"]
core_ind = [c for c in ind_cols if src_of.get(c) in LAB_CORE]
other_ind = [c for c in ind_cols if c not in core_ind]
corr = df[ind_cols].corr().abs()
for c in core_ind:
    m = corr.loc[c, other_ind].drop(labels=[x for x in [c] if x in other_ind], errors="ignore")
    top = m.sort_values(ascending=False).head(1)
    print(f"  {c:26s} max |corr| with another indicator = {top.iloc[0]:.2f}  ({top.index[0]})")

print("\nVERDICT RULE:")
print("  lab-core summed importance ~ 0  -> null REAL (genuinely unimportant).")
print("  lab-core summed importance >> non-SOFA-per-feature scale -> redundancy ARTIFACT;")
print("  corroborate with the correlation screen (high |corr| = substitutable).")
print("wrote: day4_permimp.csv")
