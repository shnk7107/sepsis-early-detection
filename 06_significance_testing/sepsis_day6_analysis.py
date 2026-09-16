"""
Day 6 · significance analysis (Nadeau-Bengio corrected).

Reads day6_repeated_perfold.csv (25 estimates/cell) and tests, per condition, per
ablation arm vs full, whether dAUPRC differs from zero.

WHY A CORRECTED TEST: repeated-CV folds reuse the same data, so the 25 deltas are
positively correlated. A naive paired t-test treats them as independent and badly
OVERSTATES significance. The Nadeau-Bengio (2003) corrected resampled t-test
inflates the variance by the train/test overlap:

    t = mean(d) / sqrt( (1/J + rho) * var(d) ),   rho = n_test/n_train,  df = J-1

For 5-fold, rho = 1/(k-1) = 0.25, J = 25  -> correction factor 0.29 vs naive 0.04.
We report BOTH (naive and corrected) so the inflation is visible; the corrected
p-value is the one to quote.

Outputs: day6_significance.csv, day6_gradient_powered.png
"""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PERFOLD = "day6_repeated_perfold.csv"
K = 5
RHO = 1.0 / (K - 1)            # n_test/n_train for k-fold = 0.25

p = pd.read_csv(PERFOLD)
J = p.groupby(["arm", "condition"]).size().max()      # 25
order = sorted(p.condition.unique(), key=lambda s: int(s.split("_")[0]))
arms_present = set(p.arm.unique())

def vec(arm, c):
    return p[(p.arm == arm) & (p.condition == c)].sort_values("split").auprc.to_numpy()

def nb_test(d):
    """Nadeau-Bengio corrected resampled t-test on paired diffs d (len J)."""
    m = d.mean(); v = d.var(ddof=1)
    se_corr = np.sqrt((1.0 / len(d) + RHO) * v)
    t = m / se_corr if se_corr > 0 else np.nan
    p_corr = 2 * stats.t.sf(abs(t), df=len(d) - 1)
    ci = stats.t.ppf(0.975, df=len(d) - 1) * se_corr     # half-width of 95% CI
    p_naive = stats.ttest_1samp(d, 0).pvalue
    return m, se_corr, p_corr, p_naive, ci

rows = []
for arm in [a for a in ["lab_panel", "lab_indicators_only", "all_7"] if a in arms_present]:
    for c in order:
        f, a = vec("full", c), vec(arm, c)
        if len(f) != J or len(a) != J:
            continue
        d = f - a
        m, se, pc, pn, ci = nb_test(d)
        rows.append(dict(arm=arm, condition=c, mean_d=m, ci95_halfwidth=ci,
                         p_corrected=pc, p_naive=pn,
                         sig_corrected=pc < 0.05, sig_naive=pn < 0.05))
res = pd.DataFrame(rows); res.to_csv("day6_significance.csv", index=False)

pd.set_option("display.width", 140, "display.float_format", lambda x: f"{x:.4f}")
for arm in [a for a in ["lab_indicators_only", "all_7", "lab_panel"] if a in arms_present]:
    t = res[res.arm == arm]
    tag = {"lab_indicators_only": "LAB CIRCULARITY (ordering beyond value)",
           "all_7": "VITAL-PROXY reliance", "lab_panel": "TOTAL lab contribution"}[arm]
    print(f"\n=== full vs {arm} · {tag} ===")
    print(t[["condition", "mean_d", "ci95_halfwidth", "p_corrected", "p_naive",
             "sig_corrected"]].to_string(index=False))
    print(f"  significant (corrected) in {int(t.sig_corrected.sum())}/8  |  "
          f"naive would claim {int(t.sig_naive.sum())}/8  |  "
          f"mean dAUPRC {t.mean_d.mean():+.4f}")

# ---- powered gradient (corrected 95% CI error bars) ----
xl = [c.split("_", 1)[1] for c in order]; x = np.arange(len(order))
colors = {"lab_panel": "#6a0dad", "lab_indicators_only": "#e8a30c", "all_7": "#0d6e8c"}
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.axhline(0, color="#888", lw=1)
for arm in [a for a in ["lab_panel", "all_7", "lab_indicators_only"] if a in arms_present]:
    t = res[res.arm == arm].set_index("condition").reindex(order)
    ax.errorbar(x, t.mean_d, yerr=t.ci95_halfwidth, fmt="o-", lw=2, capsize=3, ms=5,
                color=colors.get(arm, "#444"), label=arm)
ax.set_xticks(x); ax.set_xticklabels(xl, rotation=30, ha="right")
ax.set_ylabel("dAUPRC = AUPRC(full) - AUPRC(ablated)")
ax.set_title("Day 6 - powered: dAUPRC with Nadeau-Bengio 95% CI (25-fold repeated CV)",
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9, title="ablation depth")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.text(0.5, -0.02, "Error bars = corrected 95% CI. A bar clearing 0 = significant "
         "ordering/reliance effect after accounting for fold correlation.",
         ha="center", fontsize=8.5, color="#555")
fig.tight_layout(); fig.savefig("day6_gradient_powered.png", bbox_inches="tight",
                                facecolor="white", dpi=130)
print("\nwrote: day6_significance.csv, day6_gradient_powered.png")
