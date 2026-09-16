"""
Day 4 · paired test  (RUN THIRD, after day4_perfold.csv exists).

Tests whether dAUPRC = AUPRC(full) - AUPRC(ablated) differs from zero, using the
5 matched per-fold values (folds are identical across arms, so the deltas are paired).

Per condition, per ablation arm (lab_core headline, all_7 companion):
  - mean and sd of the 5 paired fold deltas
  - paired t-test (ttest_rel) and Wilcoxon signed-rank p-values
  - sign consistency (all 5 folds same direction)

HONEST POWER CAVEAT: with k=5 folds there are only 5 paired observations per
condition, so individual p-values are underpowered. The robust readout is the
COMBINATION of (a) mean delta magnitude vs sd, (b) sign consistency across folds,
and (c) the pattern across the ordered gradient. For publication-grade significance,
upgrade day4_perfold.py to RepeatedStratifiedKFold (e.g. 5x5=25 estimates) on a
multi-core machine and re-run; this script works unchanged on the richer file.

Output: day4_paired_results.csv + console.
"""
import numpy as np, pandas as pd
from scipy import stats

PERFOLD = "day4_perfold.csv"
OUT = "day4_paired_results.csv"

p = pd.read_csv(PERFOLD)
order = sorted(p.condition.unique(), key=lambda s: int(s.split("_")[0]))

def fold_vec(arm, cond):
    s = p[(p.arm == arm) & (p.condition == cond)].sort_values("fold")
    return s.auprc.to_numpy()

rows = []
for tag, ablated in [("lab_core", "lab_core"), ("all_7", "all_7")]:
    for c in order:
        full = fold_vec("full", c)
        abl = fold_vec(ablated, c)
        if len(full) != 5 or len(abl) != 5:
            continue                      # cell not yet computed
        d = full - abl                    # paired per-fold delta
        t_p = stats.ttest_rel(full, abl).pvalue
        try:
            w_p = stats.wilcoxon(full, abl).pvalue
        except ValueError:
            w_p = np.nan                  # zeros / ties
        rows.append(dict(ablation=tag, condition=c,
                         mean_d=d.mean(), sd_d=d.std(ddof=1),
                         ttest_p=t_p, wilcoxon_p=w_p,
                         all_same_sign=bool(np.all(d > 0) or np.all(d < 0))))

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)

pd.set_option("display.width", 130, "display.float_format", lambda x: f"{x:.4f}")
for tag in ["lab_core", "all_7"]:
    t = res[res.ablation == tag]
    head = "headline" if tag == "lab_core" else "companion"
    print(f"\n=== paired dAUPRC test · {tag} ({head}) ===")
    print(t[["condition", "mean_d", "sd_d", "ttest_p", "wilcoxon_p", "all_same_sign"]].to_string(index=False))
    sig = t[(t.ttest_p < 0.05)]
    print(f"  conditions with t-test p<0.05: {len(sig)}/{len(t)}  |  "
          f"mean dAUPRC across gradient: {t.mean_d.mean():+.4f}  |  "
          f"sign-consistent: {int(t.all_same_sign.sum())}/{len(t)}")
print("\nwrote:", OUT)
