"""
PS-A (matched) full analysis. Run AFTER psA_matched_run.py finishes.
Inputs  : psA_crosssite_preds.csv, psA_withinsrc_perfold.csv
Outputs : psA_Q1_ordering.csv          (ordering reliance, cross-site, bootstrap CIs)
          psA_Q2_resampling_transfer.csv(SMOTENC penalty within vs cross + DiD)
          psA_context_gap.csv           (transportability gap)
          psA_analysis_report.txt       (the full printed report, saved)
Point the paths at your OUT_DIR if the CSVs are on Drive.
"""
import pandas as pd, numpy as np, sys, io
from sklearn.metrics import average_precision_score

CROSS  = "psA_crosssite_preds.csv"
WITHIN = "psA_withinsrc_perfold.csv"
NB     = 2000
RNG    = np.random.default_rng(42)
SRC    = {"A->B": "A", "B->A": "B"}

cross  = pd.read_csv(CROSS)
within = pd.read_csv(WITHIN)
models = ["RF", "xgboost", "lightgbm"]
dirs   = ["A->B", "B->A"]

# capture everything printed into a report file too
buf = io.StringIO()
def out(*a):
    s = " ".join(str(x) for x in a); print(s); buf.write(s + "\n")

def cross_auprc(direction, model, cond, arm):
    s = cross[(cross.direction==direction)&(cross.model==model)&(cross.condition==cond)&(cross.arm==arm)]
    return average_precision_score(s.y_true.values, s.prob.values)

def within_mean(site, model, cond, arm):
    w = within[(within.site==site)&(within.model==model)&(within.condition==cond)&(within.arm==arm)]
    return w.auprc.mean()

def boot_contrast(direction, model, cellA, cellB):
    """paired target bootstrap of AUPRC(cellA) - AUPRC(cellB) on the same target patients."""
    def get(cond, arm):
        s = cross[(cross.direction==direction)&(cross.model==model)&(cross.condition==cond)&(cross.arm==arm)]
        return s.set_index("pid")[["y_true","prob"]]
    a = get(*cellA); b = get(*cellB); j = a.join(b, lsuffix="_a", rsuffix="_b")
    y = j.y_true_a.values; pa = j.prob_a.values; pb = j.prob_b.values
    obs = average_precision_score(y, pa) - average_precision_score(y, pb)
    n = len(y); d = np.empty(NB)
    for i in range(NB):
        ix = RNG.integers(0, n, n)
        d[i] = np.nan if y[ix].sum()==0 else average_precision_score(y[ix], pa[ix]) - average_precision_score(y[ix], pb[ix])
    return obs, np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)

# ---------- Q1 : ordering reliance ----------
out("="*104)
out("Q1  ORDERING RELIANCE cross-site. delta = AUPRC(full) - AUPRC(ordering_dropped)")
out("    <0 & CI excludes 0 => dropping ordering IMPROVES transfer (shortcut). Paired target bootstrap.")
out("="*104)
q1 = []
for cond in ["1_none", "5_SMOTENC"]:
    for direction in dirs:
        for model in models:
            obs, lo, hi = boot_contrast(direction, model, (cond,"full"), (cond,"lab_indicators_only"))
            q1.append(dict(cond=cond, direction=direction, model=model, delta=obs, lo=lo, hi=hi, sig=(lo>0 or hi<0)))
q1 = pd.DataFrame(q1)
out(q1.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
out(f"\n  mean ordering delta = {q1.delta.mean():+.4f} | cells with CI excluding 0: {q1.sig.sum()}/{len(q1)}\n")

# ---------- Q2 : resampling x transport ----------
out("="*104)
out("Q2  RESAMPLING x TRANSPORT. SMOTENC penalty = AUPRC(1_none) - AUPRC(5_SMOTENC)  (>0 = SMOTENC hurts)")
out("    DiD = cross_penalty - within_penalty  (>0 = SMOTENC hurts MORE across sites). arm=full.")
out("="*104)
q2 = []
for direction in dirs:
    site = SRC[direction]
    for model in models:
        cross_pen  = cross_auprc(direction,model,"1_none","full") - cross_auprc(direction,model,"5_SMOTENC","full")
        within_pen = within_mean(site,model,"1_none","full")      - within_mean(site,model,"5_SMOTENC","full")
        _, lo, hi = boot_contrast(direction, model, ("1_none","full"), ("5_SMOTENC","full"))
        q2.append(dict(direction=direction, model=model, within_pen=within_pen, cross_pen=cross_pen,
                       DiD=cross_pen-within_pen, cross_pen_lo=lo, cross_pen_hi=hi, cross_sig=(lo>0)))
q2 = pd.DataFrame(q2)
out(q2.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
out(f"\n  mean within SMOTENC penalty = {q2.within_pen.mean():+.4f} | mean cross = {q2.cross_pen.mean():+.4f}")
out(f"  mean transfer-penalty DiD   = {q2.DiD.mean():+.4f} | cross penalty CI excludes 0: {q2.cross_sig.sum()}/{len(q2)}\n")

# ---------- context : transportability gap ----------
out("="*104)
out("CONTEXT  transportability gap = within-source AUPRC - cross-site AUPRC (arm=full, 1_none)")
out("="*104)
ctx = []
for direction in dirs:
    site = SRC[direction]
    for model in models:
        wn = within_mean(site,model,"1_none","full"); cx = cross_auprc(direction,model,"1_none","full")
        ctx.append(dict(direction=direction, model=model, within=wn, cross=cx, gap=wn-cx, pct_drop=100*(wn-cx)/wn))
ctx = pd.DataFrame(ctx)
out(ctx.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
out(f"\n  mean in-distribution AUPRC={ctx.within.mean():.3f} -> cross-site={ctx.cross.mean():.3f} (mean drop {ctx.pct_drop.mean():.0f}%)")

q1.to_csv("psA_Q1_ordering.csv", index=False)
q2.to_csv("psA_Q2_resampling_transfer.csv", index=False)
ctx.to_csv("psA_context_gap.csv", index=False)
open("psA_analysis_report.txt","w").write(buf.getvalue())
print("\nSAVED: psA_Q1_ordering.csv, psA_Q2_resampling_transfer.csv, psA_context_gap.csv, psA_analysis_report.txt")
