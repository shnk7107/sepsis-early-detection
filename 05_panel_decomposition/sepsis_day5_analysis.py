"""
Day 5 · analysis — ablation depth + lab-circularity decomposition.

Reads day4_grid_results.csv (now incl. lab_panel, lab_indicators_only) and
day4_perfold.csv. Produces:
  1. dAUPRC table for each ablation depth, per condition (test set).
  2. The decomposition, per condition:
        total_lab  = full - lab_panel              (physiology + ordering)
        circularity= full - lab_indicators_only    (ordering beyond value)
        physiology = lab_panel - lab_indicators_only
  3. Overlay gradient: dAUPRC across the ordered 8 conditions for every ablation
     depth on one figure (3-core vs whole-panel vs indicators-only vs all_7).
  4. Paired test (t-test + Wilcoxon + sign consistency) on the panel arms using
     the matched per-fold deltas.

Outputs: day5_decomposition.csv, day5_gradient.png
"""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID, PERFOLD = "day4_grid_results.csv", "day4_perfold.csv"
g = pd.read_csv(GRID)
order = sorted(g.condition.unique(), key=lambda s: int(s.split("_")[0]))
def ta(a): return g[g.arm == a].set_index("condition").reindex(order)

arms_present = set(g.arm.unique())
full = ta("full")

# ---- dAUPRC per ablation depth (test) ----
depths = [a for a in ["lab_core", "lab_panel", "lab_indicators_only", "all_7"] if a in arms_present]
delta = pd.DataFrame({a: full.test_auprc - ta(a).test_auprc for a in depths}, index=order)
print("=== dAUPRC = AUPRC(full) - AUPRC(ablated), TEST set ===")
print(delta.round(4).to_string())

# ---- decomposition (needs both panel arms) ----
if {"lab_panel", "lab_indicators_only"} <= arms_present:
    lp, li = ta("lab_panel"), ta("lab_indicators_only")
    decomp = pd.DataFrame({
        "total_lab(full-panel)":        full.test_auprc - lp.test_auprc,
        "circularity(full-indicators)": full.test_auprc - li.test_auprc,
        "physiology(panel-indicators)": lp.test_auprc - li.test_auprc,
    }, index=order)
    print("\n=== lab-signal decomposition (test dAUPRC) ===")
    print(decomp.round(4).to_string())
    print(f"\nmean circularity (ordering beyond value): {decomp['circularity(full-indicators)'].mean():+.4f}")
    decomp.to_csv("day5_decomposition.csv")

# ---- paired test on the panel arms ----
p = pd.read_csv(PERFOLD)
def fv(arm, c):
    s = p[(p.arm == arm) & (p.condition == c)].sort_values("fold")
    return s.auprc.to_numpy()
print("\n=== paired dAUPRC test (per-fold) ===")
for arm in [a for a in ["lab_panel", "lab_indicators_only"] if a in arms_present]:
    print(f"\n--- full vs {arm} ---")
    for c in order:
        f, a = fv("full", c), fv(arm, c)
        if len(f) != 5 or len(a) != 5:
            print(f"  {c:16s} (per-fold missing)"); continue
        d = f - a
        try: w = stats.wilcoxon(f, a).pvalue
        except ValueError: w = np.nan
        same = bool(np.all(d > 0) or np.all(d < 0))
        print(f"  {c:16s} mean_d={d.mean():+.4f} sd={d.std(ddof=1):.4f} "
              f"t_p={stats.ttest_rel(f,a).pvalue:.3f} wilcoxon_p={w:.4f} same_sign={same}")

# ---- overlay gradient ----
xl = [c.split("_", 1)[1] for c in order]; x = np.arange(len(order))
colors = {"lab_core": "#c1121f", "lab_panel": "#6a0dad",
          "lab_indicators_only": "#e8a30c", "all_7": "#0d6e8c"}
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.axhline(0, color="#888", lw=1)
for a in depths:
    sd = np.sqrt(full.cv_auprc_sd**2 + ta(a).cv_auprc_sd**2)
    ax.errorbar(x, delta[a], yerr=sd, fmt="o-", lw=2, capsize=3, ms=5,
                color=colors.get(a, "#444"), label=a)
ax.set_xticks(x); ax.set_xticklabels(xl, rotation=30, ha="right")
ax.set_ylabel("dAUPRC = AUPRC(full) - AUPRC(ablated)")
ax.set_title("Day 5 - lab-circularity by ablation depth across the resampling gradient",
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9, title="ablation depth")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.text(0.5, -0.02, "Error bars = pooled fold-sd (approximate). lab_indicators_only "
         "isolates the ordering/circularity signal; lab_panel adds lab physiology.",
         ha="center", fontsize=8.5, color="#555")
fig.tight_layout(); fig.savefig("day5_gradient.png", bbox_inches="tight", facecolor="white", dpi=130)
print("\nwrote: day5_decomposition.csv, day5_gradient.png")
