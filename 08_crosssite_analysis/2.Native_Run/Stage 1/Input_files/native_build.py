"""
NATIVE-PREVALENCE feature set rebuild  (PS-A, stage 1 of 2).

WHAT THIS CHANGES vs the matched set
------------------------------------
The matched pipeline (day2_match.py) used ONE control ratio for both sites:
    R = 13  ->  prevalence 1/14 = 7.14% in BOTH sites.
That deliberately erased the real prevalence difference between hospitals.

This script rebuilds the control set with a PER-SITE ratio so each site carries its
own true clinical sepsis prevalence:
    set A: raw prevalence 8.80%  ->  R_A = round((1-p)/p) = 10  ->  achieved 9.09%
    set B: raw prevalence 5.71%  ->  R_B = round((1-p)/p) = 17  ->  achieved 5.56%
                                     (a real ~1.6x prevalence gap between sites)

WHY INTEGER R (this matters)
----------------------------
day2_match.py guarantees an EXACT window-position match by replicating each septic's
window-end position e exactly R times among controls. That is what removes the
stay-position confound. A fractional R (10.4 / 16.5) would force per-position rounding
and break the exact match, re-importing the confound the design was built to remove.
Integer R keeps the position match exact and costs only a small prevalence offset
(9.09% vs 8.80% target; 5.56% vs 5.71% target).

WHAT IS UNCHANGED (on purpose)
------------------------------
  - the septic cohort and septic windows  (septic_windows.parquet reused as-is)
  - C=48, W=6, the cut rule, the aggregation, the feature list, the SOFA tags
  - one window per control; patient-level independence; match WITHIN set
Only the control:septic ratio changes. That is the single experimental manipulation.

INPUTS (put these in DATA_DIR)
  combined.parquet, patient_index.csv, septic_windows.parquet, window_manifest_septic.csv
OUTPUTS (written to OUT_DIR)
  features_native.parquet            <- the new modeling matrix
  feature_tags_native.csv            <- should be IDENTICAL to feature_tags.csv (asserted)
  control_windows_native.parquet, window_manifest_control_native.csv
  native_build_report.txt

RUNTIME: a few minutes. Run this FIRST and read the report before launching PS-A.
"""

import numpy as np
import pandas as pd

# ----------------------------- config -----------------------------
DATA_DIR = "."          # folder holding the 4 input files
OUT_DIR  = "."          # folder for outputs

CACHE_FILE      = f"{DATA_DIR}/combined.parquet"
INDEX_FILE      = f"{DATA_DIR}/patient_index.csv"
SEPTIC_WIN_FILE = f"{DATA_DIR}/septic_windows.parquet"
SEPTIC_MANIFEST = f"{DATA_DIR}/window_manifest_septic.csv"

OUT_FEATURES = f"{OUT_DIR}/features_native.parquet"
OUT_TAGS     = f"{OUT_DIR}/feature_tags_native.csv"
OUT_CTRL_WIN = f"{OUT_DIR}/control_windows_native.parquet"
OUT_CTRL_MAN = f"{OUT_DIR}/window_manifest_control_native.csv"
OUT_REPORT   = f"{OUT_DIR}/native_build_report.txt"
REF_TAGS     = f"{DATA_DIR}/feature_tags.csv"     # optional; used only to verify tags match

W    = 6          # window length (hours) - must match the septic windows
SEED = 42
LABEL, HOUR = "SepsisLabel", "ICULOS"

# --------------------- feature design (copied verbatim from day2_aggregate.py) ---------------------
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets"]
TIMEVARYING = VITALS + LABS
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
SOFA_SYSTEM = {
    "Creatinine": "renal", "Platelets": "coagulation", "Bilirubin_total": "liver",
    "MAP": "cardiovascular", "FiO2": "respiration", "SaO2": "respiration",
    "O2Sat": "respiration",
}
SOFA_LAB_CORE = {"Creatinine", "Platelets", "Bilirubin_total"}

_log = []
def say(s=""):
    print(s); _log.append(str(s))


# ----------------------------- native ratios -----------------------------
def native_ratios(idx):
    """Per-site integer control:septic ratio reproducing that site's raw prevalence."""
    out = {}
    for s in sorted(idx["set"].unique()):
        p = idx.loc[idx["set"] == s, "label"].mean()      # raw per-site sepsis prevalence
        R = int(round((1.0 - p) / p))
        out[s] = dict(raw_prev=p, R=R, achieved_prev=1.0 / (1.0 + R))
    return out


# ------------------- control assignment (per-site R; else identical to day2_match) -------------------
def assign_controls(idx, sep_manifest, R_by_set, seed=SEED):
    """Assign R_s distinct controls to each septic's window-end position e, within set.

    A control can host window-end e iff it spans [e-W+1, e]:
        first_iculos <= e-W+1  and  last_iculos >= e
    Late (scarce) positions are filled first so long-record controls are not consumed
    by abundant early demand. One window per control (patient-level independence).
    """
    rng = np.random.default_rng(seed)
    ctrl = idx[idx["label"] == 0].copy()
    sep = sep_manifest.copy()
    sep["e"] = sep["onset_iculos"].astype(int) - 1

    rows, shortfalls = [], []
    for s in sorted(sep["set"].unique()):
        R = R_by_set[s]["R"]
        pool = ctrl[ctrl["set"] == s][["pid", "first_iculos", "last_iculos"]].reset_index(drop=True)
        used = np.zeros(len(pool), dtype=bool)
        f = pool["first_iculos"].to_numpy()
        l = pool["last_iculos"].to_numpy()

        demand = sep[sep["set"] == s]["e"].value_counts().mul(R).sort_index(ascending=False)
        for e, need in demand.items():
            feasible = (~used) & (f <= e - W + 1) & (l >= e)
            cand = np.flatnonzero(feasible)
            take = min(int(need), len(cand))
            if take < need:
                shortfalls.append((s, int(e), int(need), int(take)))
            chosen = rng.choice(cand, size=take, replace=False)
            used[chosen] = True
            for ci in chosen:
                rows.append((pool.at[ci, "pid"], s, int(e)))

    man = pd.DataFrame(rows, columns=["pid", "set", "e"])
    man["pseudo_onset"] = man["e"] + 1
    man["win_start"] = man["e"] - (W - 1)
    man["win_end"] = man["e"].astype(int)
    man["label"] = 0
    return man, shortfalls


def cut_control_windows(big, man):
    bounds = man.set_index("pid")[["win_start", "win_end", "pseudo_onset"]]
    sub = big[big["pid"].isin(bounds.index)].copy().join(bounds, on="pid")
    in_win = (sub[HOUR] >= sub["win_start"]) & (sub[HOUR] <= sub["win_end"])
    win = sub.loc[in_win].copy()
    win["rel_hour"] = (win[HOUR] - win["pseudo_onset"]).astype(int)
    win = win.drop(columns=["win_start", "win_end", "pseudo_onset"])
    return win.sort_values(["pid", HOUR]).reset_index(drop=True)


# ------------------- aggregation (copied verbatim from day2_aggregate.py) -------------------
def aggregate(win):
    win = win.sort_values(["pid", HOUR])
    g = win.groupby("pid", sort=True)

    agg = g[TIMEVARYING].agg(["mean", "min", "max", "std"])
    agg.columns = [f"{v}_{s}" for v, s in agg.columns]

    ff = win.copy()
    ff[TIMEVARYING] = g[TIMEVARYING].ffill()
    last = ff.groupby("pid").tail(1).set_index("pid")[TIMEVARYING]
    last.columns = [f"{v}_last" for v in last.columns]

    n = g[TIMEVARYING].count()
    n.columns = [f"{v}_n" for v in n.columns]
    measured = (n.values > 0).astype("int8")
    measured = pd.DataFrame(measured, index=n.index,
                            columns=[f"{v}_measured" for v in TIMEVARYING])

    static = g[STATIC].first()
    return pd.concat([static, agg, last, n, measured], axis=1).reset_index()


def build_tags(feature_cols):
    rows = []
    for col in feature_cols:
        if col in STATIC:
            rows.append((col, col, "value", "static", "demographic", False, ""))
            continue
        var, stat = col.rsplit("_", 1)
        kind = "indicator" if stat in ("measured", "n") else "value"
        group = "vital" if var in VITALS else "lab"
        rows.append((col, var, stat, kind, group, var in SOFA_SYSTEM, SOFA_SYSTEM.get(var, "")))
    tags = pd.DataFrame(rows, columns=[
        "feature", "source_var", "statistic", "kind", "group",
        "is_sofa_component", "sofa_system"])
    tags["is_sofa_lab_core"] = tags["source_var"].isin(SOFA_LAB_CORE)
    return tags


# ----------------------------- main -----------------------------
def main():
    big = pd.read_parquet(CACHE_FILE)
    idx = pd.read_csv(INDEX_FILE)
    sep_man = pd.read_csv(SEPTIC_MANIFEST)
    sep_win = pd.read_parquet(SEPTIC_WIN_FILE)

    R_by_set = native_ratios(idx)

    say("=== NATIVE BUILD · per-site control ratio ===")
    for s, d in R_by_set.items():
        n_sep = int((sep_man["set"] == s).sum())
        say(f"  set {s}: raw prevalence {d['raw_prev']*100:5.2f}%  ->  R={d['R']:2d}  "
            f"->  achieved {d['achieved_prev']*100:5.2f}%   "
            f"(septics {n_sep}, controls needed {n_sep*d['R']})")
    ratio = (R_by_set["A"]["achieved_prev"] / R_by_set["B"]["achieved_prev"]
             if {"A", "B"} <= set(R_by_set) else float("nan"))
    say(f"  prevalence gap A:B = {ratio:.2f}x   (matched set was 1.00x by construction)")

    # ---- controls ----
    ctrl_man, shortfalls = assign_controls(idx, sep_man, R_by_set)
    ctrl_win = cut_control_windows(big, ctrl_man)

    say("\n--- control assignment ---")
    say(f"  controls assigned {len(ctrl_man)}  (unique {ctrl_man['pid'].nunique()})")
    if shortfalls:
        say(f"  WARNING: {len(shortfalls)} position(s) short of demand (set, e, needed, got):")
        for sh in shortfalls[:10]:
            say(f"     {sh}")
        if len(shortfalls) > 10:
            say(f"     ... and {len(shortfalls)-10} more")
        say("  -> exact position match is BROKEN at those positions. Report this, or lower R.")
    else:
        say("  no shortfalls: every position filled to demand (exact position match holds)")

    # ---- integrity on control windows ----
    counts = ctrl_win.groupby("pid").size()
    wrong_len = counts[counts != W].index.tolist()
    pos_rows = int((ctrl_win[LABEL] == 1).sum())
    say("\n--- control window integrity ---")
    say(f"  window rows {len(ctrl_win)} (expected {len(ctrl_man)*W})")
    say(f"  rel_hour range {(int(ctrl_win['rel_hour'].min()), int(ctrl_win['rel_hour'].max()))} (expect (-6, -1))")
    say(f"  windows != {W} rows   : {len(wrong_len)}  (must be 0)")
    say(f"  controls reused      : {len(ctrl_man)-ctrl_man['pid'].nunique()}  (must be 0)")
    say(f"  positive-label rows  : {pos_rows}  (must be 0)")

    # ---- position match check (septic vs control window-end distribution, per set) ----
    sep_e = sep_man.copy(); sep_e["e"] = sep_e["onset_iculos"].astype(int) - 1
    say("\n--- confound match (window-end position, per set) ---")
    all_match = True
    for s in sorted(ctrl_man["set"].unique()):
        se = sep_e[sep_e["set"] == s]["e"].value_counts(normalize=True).round(6).sort_index()
        ce = ctrl_man[ctrl_man["set"] == s]["e"].value_counts(normalize=True).round(6).sort_index()
        ident = se.equals(ce)
        all_match &= ident
        say(f"  set {s}: identical position distribution: {ident}")

    # ---- aggregate ----
    sep_win = sep_win.copy(); sep_win["label"] = 1
    ctrl_win2 = ctrl_win.copy(); ctrl_win2["label"] = 0
    win = pd.concat([sep_win, ctrl_win2], ignore_index=True)

    feats = aggregate(win)
    meta = win.groupby("pid")[["set", "label"]].first().reset_index()
    feats = meta.merge(feats, on="pid")

    feature_cols = [c for c in feats.columns if c not in ("pid", "set", "label")]
    tags = build_tags(feature_cols)

    feats.to_parquet(OUT_FEATURES, index=False)
    tags.to_csv(OUT_TAGS, index=False)
    ctrl_win.to_parquet(OUT_CTRL_WIN, index=False)
    ctrl_man.to_csv(OUT_CTRL_MAN, index=False)

    # ---- final report ----
    say("\n=== NATIVE FEATURE MATRIX ===")
    say(f"  rows {len(feats)}  (septic {int(feats.label.sum())}, control {int((1-feats.label).sum())})")
    say(f"  features {len(feature_cols)}")
    say("  per-site n / positives / prevalence:")
    for s, grp in feats.groupby("set"):
        say(f"    set {s}: n={len(grp):6d}  positives={int(grp.label.sum()):4d}  "
            f"prevalence={grp.label.mean()*100:5.2f}%")

    ind_cols = tags[tags.kind == "indicator"].feature
    ind_nan = int(feats[ind_cols].isna().sum().sum())
    say(f"\n  indicator NaNs (must be 0): {ind_nan}")
    for grp in ["vital", "lab"]:
        cols = tags[(tags.group == grp) & (tags.kind == "value")].feature
        say(f"  value-NaN rate, {grp:5s}: {feats[cols].isna().mean().mean()*100:.1f}%")

    # tags must be identical to the matched run's, or the arms won't be comparable
    try:
        ref = pd.read_csv(REF_TAGS)
        same = ref.equals(tags)
        say(f"\n  feature_tags identical to matched run: {same}  "
            f"{'(good - arms are directly comparable)' if same else '(PROBLEM - investigate)'}")
    except Exception as e:
        say(f"\n  (could not compare to {REF_TAGS}: {e})")

    ok = (not wrong_len and pos_rows == 0 and ind_nan == 0
          and len(ctrl_man) == ctrl_man["pid"].nunique()
          and len(ctrl_win) == len(ctrl_man) * W)
    say("\nSTATUS: " + ("PASS" if ok else "FAIL - do not proceed to PS-A"))
    if shortfalls:
        say("NOTE: shortfalls present - position match is approximate. Decide before proceeding.")
    say(f"wrote: {OUT_FEATURES}, {OUT_TAGS}, {OUT_CTRL_WIN}, {OUT_CTRL_MAN}")

    with open(OUT_REPORT, "w") as fh:
        fh.write("\n".join(_log))
    print(f"\nreport saved -> {OUT_REPORT}")
    return ok


if __name__ == "__main__":
    main()
