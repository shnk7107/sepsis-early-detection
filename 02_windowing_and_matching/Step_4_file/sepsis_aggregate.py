"""
Day 2 · STEP 4 — aggregate each window to ONE feature row per patient.

Output is the patient-level feature matrix the models bind to: 1,127 septic +
14,651 control = 15,778 rows, one window each.

Feature design:
  time-varying vars (34): mean, min, max, last(LOCF), std   [value]
                          measured(0/1), n(0..6)            [indicator]
  static vars (5): Age, Gender, Unit1, Unit2, HospAdmTime   [as-is]
  excluded: ICULOS, rel_hour  -- ICULOS is the matched position variable;
            re-adding it would re-import the confound. Excluded on purpose.

No imputation here. Value NaNs are preserved (imputed in-fold at modeling time to
avoid leakage across the CV split). Indicators are complete by construction.

SOFA tagging drives the ablation. is_sofa_component + sofa_system are recorded per
feature so the ablation can drop the full SOFA set or the labs-only subset.
"""

import numpy as np
import pandas as pd

SEPTIC_WIN = "septic_windows.parquet"
CONTROL_WIN = "control_windows.parquet"
FEATURES_FILE = "features.parquet"
TAGS_FILE = "feature_tags.csv"

HOUR = "ICULOS"

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets"]
TIMEVARYING = VITALS + LABS
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]

VALUE_STATS = ["mean", "min", "max", "last", "std"]

# SOFA components present in PhysioNet (the circularity surface, the ablation target)
SOFA_SYSTEM = {
    "Creatinine": "renal", "Platelets": "coagulation", "Bilirubin_total": "liver",
    "MAP": "cardiovascular", "FiO2": "respiration", "SaO2": "respiration",
    "O2Sat": "respiration",
}
# the strongest circularity claim is lab-measurement-based:
SOFA_LAB_CORE = {"Creatinine", "Platelets", "Bilirubin_total"}


def aggregate(win):
    """Collapse a long window table (one row/patient-hour) to one row/patient."""
    win = win.sort_values(["pid", HOUR])
    g = win.groupby("pid", sort=True)

    # value stats (NaN-aware). std with <2 points -> NaN.
    agg = g[TIMEVARYING].agg(["mean", "min", "max", "std"])
    agg.columns = [f"{v}_{s}" for v, s in agg.columns]

    # last = last non-null in window (LOCF then take final row)
    ff = win.copy()
    ff[TIMEVARYING] = g[TIMEVARYING].ffill()
    last = ff.groupby("pid").tail(1).set_index("pid")[TIMEVARYING]
    last.columns = [f"{v}_last" for v in last.columns]

    # indicators: n (count of measurements), measured (binary)
    n = g[TIMEVARYING].count()
    n.columns = [f"{v}_n" for v in n.columns]
    measured = (n.values > 0).astype("int8")
    measured = pd.DataFrame(measured, index=n.index,
                            columns=[f"{v}_measured" for v in TIMEVARYING])

    # static: first value per patient
    static = g[STATIC].first()

    feats = pd.concat([static, agg, last, n, measured], axis=1).reset_index()
    return feats


def build_tags(feature_cols):
    """One row per feature column: source var, statistic, kind, group, SOFA tags."""
    rows = []
    for col in feature_cols:
        if col in STATIC:
            rows.append((col, col, "value", "static", "demographic", False, ""))
            continue
        var, stat = col.rsplit("_", 1)
        kind = "indicator" if stat in ("measured", "n") else "value"
        group = "vital" if var in VITALS else "lab"
        is_sofa = var in SOFA_SYSTEM
        rows.append((col, var, stat, kind, group, is_sofa, SOFA_SYSTEM.get(var, "")))
    tags = pd.DataFrame(rows, columns=[
        "feature", "source_var", "statistic", "kind", "group",
        "is_sofa_component", "sofa_system"])
    tags["is_sofa_lab_core"] = tags["source_var"].isin(SOFA_LAB_CORE)
    return tags


def main():
    sep = pd.read_parquet(SEPTIC_WIN); sep["label"] = 1
    ctl = pd.read_parquet(CONTROL_WIN); ctl["label"] = 0
    win = pd.concat([sep, ctl], ignore_index=True)

    feats = aggregate(win)
    # attach meta (label, set) — one per patient
    meta = win.groupby("pid")[["set", "label"]].first().reset_index()
    feats = meta.merge(feats, on="pid")

    feature_cols = [c for c in feats.columns if c not in ("pid", "set", "label")]
    tags = build_tags(feature_cols)

    feats.to_parquet(FEATURES_FILE, index=False)
    tags.to_csv(TAGS_FILE, index=False)

    # ---- report ----
    n_val = (tags.kind == "value").sum()
    n_ind = (tags.kind == "indicator").sum()
    n_sofa = tags.is_sofa_component.sum()
    print("=== STEP 4 · aggregation ===")
    print(f"patients (rows)       {len(feats)}  "
          f"(septic {int(feats.label.sum())}, control {int((1-feats.label).sum())})")
    print(f"features              {len(feature_cols)}  "
          f"(value {n_val}, indicator {n_ind}, static {len(STATIC)})")
    print(f"SOFA-component feats   {n_sofa}  "
          f"(lab-core {int(tags.is_sofa_lab_core.sum())})")
    print("\n--- integrity ---")
    ind_cols = tags[tags.kind == "indicator"].feature
    print(f"indicator NaNs (must be 0): {int(feats[ind_cols].isna().sum().sum())}")
    print(f"label counts: {feats.label.value_counts().to_dict()}")
    print(f"prevalence: {feats.label.mean()*100:.2f}%")

    # value-NaN rate by group (shows lab sparsity carrying into features)
    for grp in ["vital", "lab"]:
        cols = tags[(tags.group == grp) & (tags.kind == "value")].feature
        rate = feats[cols].isna().mean().mean() * 100
        print(f"value-NaN rate, {grp:5s}: {rate:.1f}%")

    ok = int(feats[ind_cols].isna().sum().sum()) == 0 and len(feats) == 15778
    print("\nSTATUS:", "PASS" if ok else "CHECK")
    print(f"wrote: {FEATURES_FILE}, {TAGS_FILE}")


if __name__ == "__main__":
    main()
