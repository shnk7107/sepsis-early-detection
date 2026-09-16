"""
Day 7 · cross-model significance — does the negative result hold across RF, XGBoost,
and LightGBM?

Reads RF from day6_repeated_perfold.csv and XGB/LGBM from day7_repeated_perfold.csv,
applies the SAME Nadeau-Bengio corrected resampled t-test as Day 6, and asks per
model: how many of the 8 resampling conditions show a corrected-significant
amplification of (a) lab circularity and (b) vital-proxy reliance.

Robustness logic: if all three models show ~chance-level corrected significance for
the circularity arm, the null is robust across model families -> strong negative
result. If one model lights up where the others don't, the null is model-dependent
and must be reported as such.

Outputs: day7_crossmodel_significance.csv, day7_crossmodel_gradient.png
"""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = 5
RHO = 1.0 / (K - 1)        # n_test/n_train

# load + tag model
rf = pd.read_csv("day6_repeated_perfold.csv"); rf["model"] = "RF"
others = pd.read_csv("day7_repeated_perfold.csv")
p = pd.concat([rf, others], ignore_index=True)
order = sorted(p.condition.unique(), key=lambda s: int(s.split("_")[0]))
models = ["RF", "xgboost", "lightgbm"]

def vec(model, arm, c):
    s = p[(p.model == model) & (p.arm == arm) & (p.condition == c)].sort_values("split")
    return s.auprc.to_numpy()

def nb(d):
    m, v = d.mean(), d.var(ddof=1)
    se = np.sqrt((1.0 / len(d) + RHO) * v)
    t = m / se if se > 0 else np.nan
    return m, se, 2 * stats.t.sf(abs(t), df=len(d) - 1), stats.t.ppf(0.975, len(d) - 1) * se

rows = []
for model in models:
    for arm in ["lab_indicators_only", "all_7", "lab_panel"]:
        for c in order:
            f, a = vec(model, "full", c), vec(model, arm, c)
            if len(f) == 0 or len(a) == 0 or len(f) != len(a):
                continue
            d = f - a; m, se, pc, ci = nb(d)
            rows.append(dict(model=model, arm=arm, condition=c, mean_d=m,
                             ci95=ci, p_corrected=pc, sig=pc < 0.05))
res = pd.DataFrame(rows); res.to_csv("day7_crossmodel_significance.csv", index=False)

print("=== corrected-significant conditions per model (out of 8) ===")
print(f"{'arm':22s}{'RF':>6}{'xgboost':>10}{'lightgbm':>11}")
for arm in ["lab_indicators_only", "all_7", "lab_panel"]:
    line = f"{arm:22s}"
    for model in models:
        t = res[(res.model == model) & (res.arm == arm)]
        line += f"{(int(t.sig.sum()) if len(t) else 0):>6}/8" if model == "RF" else f"{(int(t.sig.sum()) if len(t) else 0):>8}/8"
    print(line)

print("\n=== mean dAUPRC across the gradient, per model ===")
print(f"{'arm':22s}{'RF':>8}{'xgboost':>10}{'lightgbm':>11}")
for arm in ["lab_indicators_only", "all_7", "lab_panel"]:
    line = f"{arm:22s}"
    for model in models:
        t = res[(res.model == model) & (res.arm == arm)]
        line += f"{(t.mean_d.mean() if len(t) else np.nan):>+10.4f}"
    print(line)

# verdict on the headline (circularity arm)
print("\n=== VERDICT: lab-circularity amplification (full vs lab_indicators_only) ===")
for model in models:
    t = res[(res.model == model) & (res.arm == "lab_indicators_only")]
    if len(t):
        print(f"  {model:9s}: {int(t.sig.sum())}/8 corrected-significant, "
              f"mean dAUPRC {t.mean_d.mean():+.4f}  "
              f"-> {'NULL holds' if t.sig.sum() <= 1 else 'effect present'}")

# ---- overlay plot: circularity gradient across the three models ----
x = np.arange(len(order)); xl = [c.split("_", 1)[1] for c in order]
cmap = {"RF": "#c1121f", "xgboost": "#0d6e8c", "lightgbm": "#2a9d4a"}
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.axhline(0, color="#888", lw=1)
for model in models:
    t = res[(res.model == model) & (res.arm == "lab_indicators_only")].set_index("condition").reindex(order)
    if t.mean_d.notna().any():
        ax.errorbar(x, t.mean_d, yerr=t.ci95, fmt="o-", lw=2, capsize=3, ms=5,
                    color=cmap[model], label=model)
ax.set_xticks(x); ax.set_xticklabels(xl, rotation=30, ha="right")
ax.set_ylabel("dAUPRC  (full - lab_indicators_only)")
ax.set_title("Day 7 - lab-circularity gradient across model families (corrected 95% CI)",
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9, title="model")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.text(0.5, -0.02, "All three lines hugging 0 with CIs crossing it = the negative "
         "result (no resampling-amplified circularity) is robust across models.",
         ha="center", fontsize=8.5, color="#555")
fig.tight_layout(); fig.savefig("day7_crossmodel_gradient.png", bbox_inches="tight",
                                facecolor="white", dpi=130)
print("\nwrote: day7_crossmodel_significance.csv, day7_crossmodel_gradient.png")
