"""
Day 2 · STEP 3 — matched control windows.

Goal: give every retained septic a set of non-septic control windows that match on
window length (W) and ICULOS position (window-end e), WITHIN hospital set, so the
classifier cannot separate classes on stay-position or site instead of physiology.
Those two confounds (position, site) are leakage paths that SOFA-ablation cannot
remove, so they must be designed out here.

Design (locked):
  - Distributional match, not 1:1 — the imbalance is the study's independent
    variable; balancing it here would destroy the resampling gradient.
  - One window per control (patient-level independence; SMOTE validity).
  - Match WITHIN set; control set-composition mirrors the septic one.
  - Imbalance anchored to true clinical prevalence (7.27%), via R=13 controls per
    septic (=> 7.14% prevalence). Each septic's exact window-end position is
    replicated R times across controls, so the control position distribution is
    identical to the septic one by construction.

R is THE study parameter (the resampling baseline). It is one constant; the natural
pool ratio (~33:1) is the documented sensitivity alternative.
"""

import numpy as np
import pandas as pd

CACHE_FILE = "combined.parquet"
INDEX_FILE = "patient_index.csv"
SEPTIC_MANIFEST = "window_manifest_septic.csv"

CONTROL_WIN_FILE = "control_windows.parquet"
CONTROL_MANIFEST = "window_manifest_control.csv"

W = 6          # window length (hours)  — must match step 2
R = 13         # controls per septic (imbalance baseline; 1/(1+R) = 7.14% prevalence)
SEED = 42

LABEL = "SepsisLabel"
HOUR = "ICULOS"


def assign_controls(idx, sep_manifest, R=R, seed=SEED):
    """Assign R distinct controls to each septic's window-end position e, within set.

    A control can host window-end e iff it spans [e-W+1, e]:
    first_iculos <= e-W+1  and  last_iculos >= e.
    Late (scarce) positions are filled first so the long-record controls they need
    are not consumed by abundant early demand.
    """
    rng = np.random.default_rng(seed)
    ctrl = idx[idx["label"] == 0].copy()
    sep = sep_manifest.copy()
    sep["e"] = sep["onset_iculos"].astype(int) - 1   # window-end ICULOS

    rows, shortfalls = [], []
    for s in sorted(sep["set"].unique()):
        pool = ctrl[ctrl["set"] == s][["pid", "first_iculos", "last_iculos"]].reset_index(drop=True)
        used = np.zeros(len(pool), dtype=bool)
        f = pool["first_iculos"].to_numpy()
        l = pool["last_iculos"].to_numpy()

        # demand per position e (each septic at e contributes R)
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
    if shortfalls:
        print("WARNING shortfalls (set, e, needed, got):", shortfalls)
    return man


def cut_control_windows(big, man):
    """Extract the W rows per assigned control. rel_hour = ICULOS - pseudo_onset."""
    bounds = man.set_index("pid")[["win_start", "win_end", "pseudo_onset"]]
    sub = big[big["pid"].isin(bounds.index)].copy().join(bounds, on="pid")
    in_win = (sub[HOUR] >= sub["win_start"]) & (sub[HOUR] <= sub["win_end"])
    win = sub.loc[in_win].copy()
    win["rel_hour"] = (win[HOUR] - win["pseudo_onset"]).astype(int)
    win = win.drop(columns=["win_start", "win_end", "pseudo_onset"])
    return win.sort_values(["pid", HOUR]).reset_index(drop=True)


def validate(win, man, sep_manifest):
    counts = win.groupby("pid").size()
    sep = sep_manifest.copy()
    sep["e"] = sep["onset_iculos"].astype(int) - 1

    checks = {
        "n_controls": int(man.shape[0]),
        "unique_controls": int(man["pid"].nunique()),
        "n_window_rows": int(win.shape[0]),
        "expected_rows": int(man.shape[0] * W),
        "windows_wrong_length": counts[counts != W].index.tolist(),
        "positive_label_rows": int((win[LABEL] == 1).sum()),
        "rel_hour_range": (int(win["rel_hour"].min()), int(win["rel_hour"].max())),
    }
    # position match per set: compare septic vs control window-end distributions
    pos = {}
    for s in sorted(man["set"].unique()):
        se = sep[sep["set"] == s]["e"]
        ce = man[man["set"] == s]["e"]
        pos[s] = {
            "septic_median": int(se.median()), "control_median": int(ce.median()),
            "septic_mean": round(float(se.mean()), 2), "control_mean": round(float(ce.mean()), 2),
            "identical_distribution": bool(
                se.value_counts(normalize=True).round(6).sort_index().equals(
                    ce.value_counts(normalize=True).round(6).sort_index())),
        }
    checks["position_match_by_set"] = pos
    # set composition: septic vs control
    checks["set_frac_septic"] = sep["set"].value_counts(normalize=True).round(3).to_dict()
    checks["set_frac_control"] = man["set"].value_counts(normalize=True).round(3).to_dict()
    return checks


def main():
    big = pd.read_parquet(CACHE_FILE)
    idx = pd.read_csv(INDEX_FILE)
    sep_man = pd.read_csv(SEPTIC_MANIFEST)

    ctrl_man = assign_controls(idx, sep_man)
    ctrl_win = cut_control_windows(big, ctrl_man)
    checks = validate(ctrl_win, ctrl_man, sep_man)

    ctrl_win.to_parquet(CONTROL_WIN_FILE, index=False)
    ctrl_man.to_csv(CONTROL_MANIFEST, index=False)

    print("=== STEP 3 · matched control windows ===")
    print(f"W={W}  R={R}  ->  prevalence {1/(1+R)*100:.2f}%")
    print(f"controls assigned     {checks['n_controls']} "
          f"(unique {checks['unique_controls']})")
    print(f"window rows           {checks['n_window_rows']} "
          f"(expected {checks['expected_rows']})")
    print(f"rel_hour range        {checks['rel_hour_range']} (expect (-6, -1))")
    print("\n--- integrity ---")
    print(f"windows != {W} rows     : {len(checks['windows_wrong_length'])} (must be 0)")
    print(f"controls reused        : "
          f"{checks['n_controls'] - checks['unique_controls']} (must be 0)")
    print(f"positive-label rows    : {checks['positive_label_rows']} (must be 0)")
    print("\n--- confound match ---")
    for s, p in checks["position_match_by_set"].items():
        print(f"set {s}: septic e median {p['septic_median']} / control {p['control_median']}"
              f"  | identical dist: {p['identical_distribution']}")
    print(f"set frac  septic {checks['set_frac_septic']}  control {checks['set_frac_control']}")

    ok = (not checks["windows_wrong_length"]
          and checks["n_controls"] == checks["unique_controls"]
          and checks["positive_label_rows"] == 0
          and checks["n_window_rows"] == checks["expected_rows"]
          and all(p["identical_distribution"] for p in checks["position_match_by_set"].values()))
    print("\nSTATUS:", "PASS" if ok else "FAIL")
    print(f"wrote: {CONTROL_WIN_FILE}, {CONTROL_MANIFEST}")
    return ok


if __name__ == "__main__":
    main()
