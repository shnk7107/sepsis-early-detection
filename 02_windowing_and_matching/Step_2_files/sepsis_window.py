"""
Day 2 windowing for the PhysioNet 2019 sepsis study.

Locked parameters (Day-2 overlap map):
  C = 48   onset-position cutoff (ICULOS). Beyond this, controls cannot host a
           matched window (supply:demand < 1 past ~58h). Restricts the design to
           early-to-mid-stay onset; later-onset sepsis is out of scope.
  W = 6    fixed window length, in hours, immediately before onset.

Locked cut rule (Day-2 decision 1):
  Features come ONLY from rows strictly before the first positive label:
  window = rows with ICULOS in [onset_iculos - W, onset_iculos - 1].
  This removes label LEAKAGE. It does not remove circularity: the pre-onset
  window still carries rising SOFA-component signal by design. That residual is
  what the ablation measures.

STEP 2 (this file, so far): cut and validate the SEPTIC windows.
  Steps 3 (matched controls) and 4 (aggregation) extend this module later.
"""

import numpy as np
import pandas as pd

CACHE_FILE = "combined.parquet"
INDEX_FILE = "patient_index.csv"
SEPTIC_WIN_FILE = "septic_windows.parquet"
SEPTIC_MANIFEST = "window_manifest_septic.csv"

C = 48   # onset-position cutoff (ICULOS hours)
W = 6    # fixed window length (hours)

LABEL = "SepsisLabel"
HOUR = "ICULOS"


# ----------------------------------------------------------------------------
# Select the retained septic cohort
# ----------------------------------------------------------------------------
def retained_septics(idx):
    """Septics inside common support with a full W-hour pre-onset window.

    Retained iff: label==1 AND onset_iculos <= C AND (onset - first) >= W.
    Returns a manifest with the window bounds per patient.
    """
    sep = idx[idx["label"] == 1].copy()
    sep["pre_hours"] = sep["onset_iculos"] - sep["first_iculos"]
    keep = sep[(sep["onset_iculos"] <= C) & (sep["pre_hours"] >= W)].copy()

    keep["win_start"] = (keep["onset_iculos"] - W).astype(int)   # inclusive
    keep["win_end"] = (keep["onset_iculos"] - 1).astype(int)     # inclusive
    return keep[["pid", "set", "label", "onset_iculos",
                 "first_iculos", "last_iculos", "win_start", "win_end"]]


# ----------------------------------------------------------------------------
# Cut the window rows out of the combined cache
# ----------------------------------------------------------------------------
def cut_septic_windows(big, manifest):
    """Extract the W rows per retained septic. Keyed off ICULOS values."""
    bounds = manifest.set_index("pid")[["win_start", "win_end", "onset_iculos"]]
    sub = big[big["pid"].isin(bounds.index)].copy()

    # join each row's window bounds, then keep rows inside [win_start, win_end]
    sub = sub.join(bounds, on="pid")
    in_win = (sub[HOUR] >= sub["win_start"]) & (sub[HOUR] <= sub["win_end"])
    win = sub.loc[in_win].copy()

    # rel_hour: -W .. -1  (hours before onset). Position within the window.
    win["rel_hour"] = (win[HOUR] - win["onset_iculos"]).astype(int)
    win = win.drop(columns=["win_start", "win_end", "onset_iculos"])
    return win.sort_values(["pid", HOUR]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Integrity: the step is only valid if these hold
# ----------------------------------------------------------------------------
def validate(win, manifest):
    """Return a dict of checks. Empty failure lists == clean."""
    counts = win.groupby("pid").size()

    leak_rows = int((win[LABEL] == 1).sum())            # MUST be 0
    wrong_len = counts[counts != W].index.tolist()      # MUST be empty
    missing = set(manifest["pid"]) - set(counts.index)  # patients with no rows
    rel_range = (int(win["rel_hour"].min()), int(win["rel_hour"].max()))

    return {
        "n_septic_windows": int(manifest.shape[0]),
        "n_window_rows": int(win.shape[0]),
        "expected_rows": int(manifest.shape[0] * W),
        "label_positive_rows_in_window": leak_rows,
        "windows_wrong_length": wrong_len,
        "manifest_pids_without_rows": sorted(missing),
        "rel_hour_range": rel_range,
    }


def main():
    big = pd.read_parquet(CACHE_FILE)
    idx = pd.read_csv(INDEX_FILE)

    manifest = retained_septics(idx)
    win = cut_septic_windows(big, manifest)
    checks = validate(win, manifest)

    win.to_parquet(SEPTIC_WIN_FILE, index=False)
    manifest.to_csv(SEPTIC_MANIFEST, index=False)

    print("=== STEP 2 · septic window cut ===")
    print(f"C={C}  W={W}")
    print(f"retained septics       {checks['n_septic_windows']}")
    print(f"window rows            {checks['n_window_rows']} "
          f"(expected {checks['expected_rows']})")
    print(f"rel_hour range         {checks['rel_hour_range']} (expect (-6, -1))")
    print("\n--- integrity (all must be clean) ---")
    print(f"positive label rows in any window : "
          f"{checks['label_positive_rows_in_window']}  (must be 0)")
    print(f"windows with != {W} rows            : "
          f"{len(checks['windows_wrong_length'])}  (must be 0)")
    print(f"retained pids with no rows         : "
          f"{len(checks['manifest_pids_without_rows'])}  (must be 0)")

    ok = (checks["label_positive_rows_in_window"] == 0
          and not checks["windows_wrong_length"]
          and not checks["manifest_pids_without_rows"]
          and checks["n_window_rows"] == checks["expected_rows"])
    print("\nSTATUS:", "PASS" if ok else "FAIL — do not proceed to step 3")
    print(f"wrote: {SEPTIC_WIN_FILE}, {SEPTIC_MANIFEST}")
    return ok


if __name__ == "__main__":
    main()
