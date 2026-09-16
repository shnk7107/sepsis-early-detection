"""
Day 4 · analysis of the resampling x ablation grid.

Reads day4_grid_results.csv (24 cells) and produces:
  1. dAUPRC table   : AUPRC(full) - AUPRC(ablated) per condition,
                      for lab_core (headline) and all_7 (companion),
                      on TEST and on CV-mean.
  2. noise-band screen : |dAUPRC| vs pooled fold-sd. APPROXIMATE -- the CSV holds
                      only per-cell mean/sd, not per-fold scores, so this is a
                      screen, not a formal paired test. A delta whose error bar
                      crosses 0 is indistinguishable from zero here.
  3. gradient plot  : dAUPRC across the ordered 8 conditions, with +/- pooled-sd
                      error bars and a y=0 line.

Outputs: day4_delta_table.csv, day4_gradient.png
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "day4_grid_results.csv"
DELTA_TABLE = "day4_delta_table.csv"
PLOT_FILE = "day4_gradient.png"

r = pd.read_csv(RESULTS_FILE)
# clean ordering: conditions sort by their leading number ("1_none" ... "8_...")
order = sorted(r.condition.unique(), key=lambda s: int(s.split("_")[0]))

def arm(a):  # indexed by condition
    return r[r.arm == a].set_index("condition")

full, lab, all7 = arm("full"), arm("lab_core"), arm("all_7")

def deltas(ablated, tag):
    out = []
    for c in order:
        d_test = full.loc[c, "test_auprc"] - ablated.loc[c, "test_auprc"]
        d_cv   = full.loc[c, "cv_auprc"]   - ablated.loc[c, "cv_auprc"]
        # pooled fold-sd of the difference (independence approximation -> conservative)
        sd_pool = np.sqrt(full.loc[c, "cv_auprc_sd"]**2 + ablated.loc[c, "cv_auprc_sd"]**2)
        out.append(dict(condition=c, ablation=tag,
                        full_test=full.loc[c, "test_auprc"],
                        ablated_test=ablated.loc[c, "test_auprc"],
                        dAUPRC_test=d_test, dAUPRC_cv=d_cv,
                        pooled_sd=sd_pool,
                        within_noise=abs(d_test) < sd_pool))
    return pd.DataFrame(out)

tab = pd.concat([deltas(lab, "lab_core"), deltas(all7, "all_7")], ignore_index=True)
tab.to_csv(DELTA_TABLE, index=False)

# ---- console report ----
pd.set_option("display.width", 120, "display.float_format", lambda x: f"{x:.3f}")
print("=== dAUPRC = AUPRC(full) - AUPRC(ablated) ===  (TEST set)\n")
for tag in ["lab_core", "all_7"]:
    t = tab[tab.ablation == tag]
    print(f"--- ablation: {tag} (headline)" if tag == "lab_core"
          else f"--- ablation: {tag} (companion)")
    print(t[["condition", "full_test", "ablated_test", "dAUPRC_test",
             "pooled_sd", "within_noise"]].to_string(index=False))
    n_null = int(t.within_noise.sum())
    print(f"    {n_null}/8 conditions have |dAUPRC| < pooled fold-sd "
          f"(indistinguishable from 0 on this screen)")
    print(f"    max |dAUPRC| = {t.dAUPRC_test.abs().max():.3f}  "
          f"| mean dAUPRC = {t.dAUPRC_test.mean():+.3f}\n")

# ---- gradient plot ----
xlabels = [c.split("_", 1)[1] for c in order]
x = np.arange(len(order))
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.axhline(0, color="#888", lw=1, ls="-")
for tag, col, off in [("lab_core", "#c1121f", -0.08), ("all_7", "#0d6e8c", 0.08)]:
    t = tab[tab.ablation == tag].set_index("condition").loc[order]
    ax.errorbar(x + off, t.dAUPRC_test, yerr=t.pooled_sd, fmt="o-", color=col,
                lw=2, capsize=4, ms=6,
                label=f"{tag} ablation  (dAUPRC +/- pooled fold-sd)")
ax.set_xticks(x); ax.set_xticklabels(xlabels, rotation=30, ha="right")
ax.set_ylabel("dAUPRC  =  AUPRC(full) - AUPRC(ablated)")
ax.set_title("Day 4 - SOFA-component reliance across the resampling gradient",
             fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.text(0.5, -0.02,
         "Points whose error bar crosses 0 are indistinguishable from zero "
         "(approximate screen; formal paired test needs per-fold scores).",
         ha="center", fontsize=8.5, color="#555")
fig.tight_layout()
fig.savefig(PLOT_FILE, bbox_inches="tight", facecolor="white", dpi=130)
print(f"wrote: {DELTA_TABLE}, {PLOT_FILE}")
