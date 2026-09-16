"""
Day 1 loader for the PhysioNet 2019 sepsis dataset.

Locked design decisions (do not drift without re-opening the study design):
  - PATIENT-LEVEL task. Each patient collapses to ONE label. The positive-hour
    block is used only to (a) confirm sepsis and (b) mark onset, so a window can
    later be cut before it. Positive hours are never prediction targets.
  - ICULOS is the hour column. All timing keys off ICULOS *values*, never row
    position: ~22% of patients enter mid-stay (first ICULOS > 1).
  - Raw .psv is the source of truth. A merged columnar cache is only a fast
    read-path; it is regenerable and never hand-edited.
  - The A/B hospital-system split is preserved via the `set` column. Combined
    into one file, never into one undifferentiated pool (domain-shift analysis
    depends on it).
  - Integrity checks return flagged-pid LISTS, never a crash. A corrupt patient
    is named; the run continues.

Artifacts produced:
  combined.parquet    full merge, one row per patient-hour (read-path)
  patient_index.csv   reduction, one row per patient (the real Day-1 interface)
  slice_pids.json     frozen correctness slice (seed + prevalence + pid list)
  eda_summary.json    EDA numbers + integrity flags
"""

import os
import glob
import json
import time

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_ROOT = "sepsis_data"
SETS = {"A": "training_setA", "B": "training_setB"}

CACHE_FILE = "combined.parquet"
INDEX_FILE = "patient_index.csv"
SLICE_FILE = "slice_pids.json"
EDA_FILE = "eda_summary.json"

# Patient-level sepsis prevalence in the full set. This is the PATIENT rate.
# It is NOT the hourly rate (~1.80%). The correctness slice is stratified to
# this value because we sample whole patients.
PATIENT_PREVALENCE = 0.0727

SLICE_SIZE = 400      # >=400 so missingness/per-class patterns are inspectable
SLICE_SEED = 42       # frozen; the pid list is persisted so re-runs are identical
MAX_POS_HOURS = 10    # expected upper bound on the positive-label block length

LABEL = "SepsisLabel"
HOUR = "ICULOS"
META_COLS = ["pid", "set"]


# ----------------------------------------------------------------------------
# 1. Raw ingestion  (source of truth)
# ----------------------------------------------------------------------------
def ingest_raw():
    """Read every .psv, tag with pid + set, concatenate. One row per patient-hour."""
    frames = []
    for set_name, subdir in SETS.items():
        paths = sorted(glob.glob(os.path.join(DATA_ROOT, subdir, "*.psv")))
        for p in paths:
            pid = os.path.splitext(os.path.basename(p))[0]
            df = pd.read_csv(p, sep="|")
            df["pid"] = pid
            df["set"] = set_name
            frames.append(df)

    big = pd.concat(frames, ignore_index=True)
    del frames  # free peak memory before any further allocation

    # Downcast floats to halve memory; values are clinical, float32 precision is ample.
    float_cols = big.select_dtypes("float64").columns
    big[float_cols] = big[float_cols].astype("float32")
    big["set"] = big["set"].astype("category")
    return big


def load_combined(force_rebuild=False):
    """Return the merged table. Read the cache if present, else ingest + cache."""
    if os.path.exists(CACHE_FILE) and not force_rebuild:
        return pd.read_parquet(CACHE_FILE)

    t0 = time.time()
    big = ingest_raw()
    big.to_parquet(CACHE_FILE, index=False)
    print(f"[ingest] {len(big):,} rows from raw .psv in {time.time() - t0:.1f}s "
          f"-> {CACHE_FILE}")
    return big


# ----------------------------------------------------------------------------
# 2. Patient index  (reduction: one row per patient)
# ----------------------------------------------------------------------------
def build_patient_index(big):
    """Collapse hourly rows to one row per patient.

    Columns: pid, set, label, onset_iculos, first_iculos, last_iculos, n_hours.
    onset_iculos is NaN for non-septic patients.
    """
    g = big.groupby("pid", sort=True)

    idx = pd.DataFrame({
        "set": g["set"].first(),
        "label": g[LABEL].max().astype("int8"),
        "first_iculos": g[HOUR].min().astype("int32"),
        "last_iculos": g[HOUR].max().astype("int32"),
        "n_hours": g.size().astype("int32"),
    })

    # onset = first ICULOS where the label is positive (NaN if never positive)
    pos = big.loc[big[LABEL] == 1]
    onset = pos.groupby("pid")[HOUR].min()
    idx["onset_iculos"] = onset.reindex(idx.index)  # NaN where no positive row

    idx = idx.reset_index()  # pid back to a column
    idx = idx[["pid", "set", "label", "onset_iculos",
               "first_iculos", "last_iculos", "n_hours"]]
    return idx


# ----------------------------------------------------------------------------
# 3. Integrity checks  (return flagged-pid lists, never raise)
# ----------------------------------------------------------------------------
def run_integrity_checks(big):
    """Per-patient integrity. Returns dict of flagged-pid lists (empty = clean).

    Checks:
      noncontiguous_iculos : ICULOS skips an hour within a patient
      duplicate_iculos     : same ICULOS appears twice within a patient
      nonmonotonic_label   : label goes 1 -> 0 (must only ever go 0 -> 1)
      excess_positive      : positive-hour count exceeds MAX_POS_HOURS
    """
    df = big[["pid", HOUR, LABEL]].sort_values(["pid", HOUR]).reset_index(drop=True)

    # Boundary mask: True on the first row of each patient (where a diff would
    # straddle two patients and must be ignored).
    boundary = df["pid"].ne(df["pid"].shift())

    d_hour = df[HOUR].diff()
    d_label = df[LABEL].diff()

    interior = ~boundary
    noncontig = df.loc[interior & (d_hour > 1), "pid"].unique().tolist()
    dup = df.loc[interior & (d_hour == 0), "pid"].unique().tolist()
    nonmono = df.loc[interior & (d_label < 0), "pid"].unique().tolist()

    pos_counts = big.groupby("pid")[LABEL].sum()
    excess = pos_counts.index[pos_counts > MAX_POS_HOURS].tolist()

    return {
        "noncontiguous_iculos": sorted(noncontig),
        "duplicate_iculos": sorted(dup),
        "nonmonotonic_label": sorted(nonmono),
        "excess_positive": sorted(excess),
    }


# ----------------------------------------------------------------------------
# 4. EDA summary  (numbers only; persisted so Day-1 state is auditable)
# ----------------------------------------------------------------------------
def compute_eda(big, idx):
    feature_cols = [c for c in big.columns if c not in META_COLS + [LABEL]]

    def dist(s):
        return {
            "min": int(s.min()), "q25": int(s.quantile(.25)),
            "median": int(s.median()), "mean": round(float(s.mean()), 2),
            "q75": int(s.quantile(.75)), "max": int(s.max()),
        }

    per_set = {}
    for s in idx["set"].cat.categories if hasattr(idx["set"], "cat") else idx["set"].unique():
        sub = idx[idx["set"] == s]
        per_set[str(s)] = {
            "patients": int(len(sub)),
            "septic": int(sub["label"].sum()),
            "patient_prevalence": round(float(sub["label"].mean()), 4),
        }

    septic = idx[idx["label"] == 1]
    pos_block = big.groupby("pid")[LABEL].sum()
    pos_block = pos_block[pos_block > 0]

    return {
        "n_patients": int(len(idx)),
        "n_rows": int(len(big)),
        "patient_prevalence": round(float(idx["label"].mean()), 4),
        "row_prevalence": round(float(big[LABEL].mean()), 4),
        "per_set": per_set,
        "n_hours_distribution": dist(idx["n_hours"]),
        "onset_iculos_septic": {
            "median": int(septic["onset_iculos"].median()),
            "mean": round(float(septic["onset_iculos"].mean()), 2),
        },
        "positive_block_length": {
            "median": int(pos_block.median()),
            "mean": round(float(pos_block.mean()), 2),
            "max": int(pos_block.max()),
            "value_counts": {int(k): int(v)
                             for k, v in pos_block.value_counts().sort_index().items()},
        },
        "missingness_pct": {
            c: round(float(big[c].isna().mean() * 100), 2) for c in feature_cols
        },
    }


# ----------------------------------------------------------------------------
# 5. Correctness slice  (frozen: seed + prevalence + pid list)
# ----------------------------------------------------------------------------
def make_slice(idx):
    """Stratified whole-patient slice matching PATIENT_PREVALENCE. Reproducible."""
    n_pos = round(SLICE_SIZE * PATIENT_PREVALENCE)
    n_neg = SLICE_SIZE - n_pos

    rng = np.random.default_rng(SLICE_SEED)
    pos_pids = idx.loc[idx["label"] == 1, "pid"].to_numpy()
    neg_pids = idx.loc[idx["label"] == 0, "pid"].to_numpy()

    pick_pos = rng.choice(pos_pids, size=n_pos, replace=False)
    pick_neg = rng.choice(neg_pids, size=n_neg, replace=False)
    pids = sorted(np.concatenate([pick_pos, pick_neg]).tolist())

    return {
        "seed": SLICE_SEED,
        "size": SLICE_SIZE,
        "target_prevalence": PATIENT_PREVALENCE,
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "pids": pids,
    }


def load_slice_from_raw(slice_spec):
    """Read the slice straight from .psv (exercises the real input path)."""
    want = set(slice_spec["pids"])
    frames = []
    for set_name, subdir in SETS.items():
        for p in sorted(glob.glob(os.path.join(DATA_ROOT, subdir, "*.psv"))):
            pid = os.path.splitext(os.path.basename(p))[0]
            if pid in want:
                df = pd.read_csv(p, sep="|")
                df["pid"] = pid
                df["set"] = set_name
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def main():
    big = load_combined()

    idx = build_patient_index(big)
    idx.to_csv(INDEX_FILE, index=False)

    flags = run_integrity_checks(big)
    eda = compute_eda(big, idx)
    eda["integrity_flags"] = {k: len(v) for k, v in flags.items()}
    eda["integrity_flagged_pids"] = flags
    with open(EDA_FILE, "w") as f:
        json.dump(eda, f, indent=2)

    slice_spec = make_slice(idx)
    with open(SLICE_FILE, "w") as f:
        json.dump(slice_spec, f, indent=2)

    # Verify the slice reads from raw and matches its declared prevalence.
    slice_df = load_slice_from_raw(slice_spec)
    slice_prev = slice_df.groupby("pid")[LABEL].max().mean()

    # ---- report ----
    print("\n=== DAY 1 REPORT ===")
    print(f"patients              {eda['n_patients']:,}")
    print(f"rows                  {eda['n_rows']:,}")
    print(f"patient prevalence    {eda['patient_prevalence']*100:.2f}%")
    print(f"row prevalence        {eda['row_prevalence']*100:.2f}%")
    print(f"set A / B prevalence  "
          f"{eda['per_set']['A']['patient_prevalence']*100:.2f}% / "
          f"{eda['per_set']['B']['patient_prevalence']*100:.2f}%")
    print(f"n_hours median/max    {eda['n_hours_distribution']['median']} / "
          f"{eda['n_hours_distribution']['max']}")
    print(f"pos-block max         {eda['positive_block_length']['max']}")
    print("integrity flags       " +
          ", ".join(f"{k}={v}" for k, v in eda["integrity_flags"].items()))
    print(f"slice                 {len(slice_spec['pids'])} pids, "
          f"prevalence {slice_prev*100:.2f}% (target "
          f"{PATIENT_PREVALENCE*100:.2f}%)")
    print(f"\nwrote: {CACHE_FILE}, {INDEX_FILE}, {SLICE_FILE}, {EDA_FILE}")


if __name__ == "__main__":
    main()
