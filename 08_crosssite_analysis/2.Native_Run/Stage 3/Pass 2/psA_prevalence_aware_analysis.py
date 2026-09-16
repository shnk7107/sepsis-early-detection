"""
PS-A analyzer, prevalence-aware.  Works for BOTH the matched and the native runs.

WHY THIS SUPERSEDES psA_full_analysis.py
----------------------------------------
The matched analyzer referenced the SOURCE site for within-site comparisons. That was
safe when both sites were forced to 7.14% prevalence. It is NOT safe on the native set,
where site A = 9.09% and site B = 5.56%: AUPRC is prevalence-dependent, so a comparison
between a score measured at 9.09% and one measured at 5.56% confounds domain shift with
a pure base-rate artifact.

Fix: every contrast is referenced to the TARGET site, so both terms are evaluated on the
same distribution at the same prevalence.
  - transportability gap = within_TARGET - cross            (the "oracle gap")
  - resampling DiD       = cross_penalty - within_TARGET_penalty
Both absolute and RELATIVE (%-of-baseline) penalties are reported; relative is the
scale-free one and is the number to trust when prevalence differs.

Q1 (ordering) needs no correction: both arms are scored on the same target patients.

HOW TO RUN
----------
Edit the three CONFIG lines below, then:  python psA_prevalence_aware_analysis.py

Run it TWICE - once for native, once for matched - so both are analyzed with identical
methodology and are directly comparable in the paper.

  native : CROSS_FILE="psA_native_crosssite_preds.csv"
           WITHIN_FILE="psA_native_withinsrc_perfold.csv"
           OUT_PREFIX="psA_native"

  matched: CROSS_FILE="psA_crosssite_preds.csv"
           WITHIN_FILE="psA_withinsrc_perfold.csv"
           OUT_PREFIX="psA_matched"

OUTPUTS: <OUT_PREFIX>_Q1_ordering.csv, <OUT_PREFIX>_Q2_resampling.csv,
         <OUT_PREFIX>_context.csv, <OUT_PREFIX>_report.txt
"""
import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score

# ------------------------- CONFIG -------------------------
CROSS_FILE  = "psA_crosssite_preds.csv"          # matched (yesterday)
WITHIN_FILE = "psA_withinsrc_perfold.csv"        # matched (yesterday)
OUT_PREFIX  = "psA_matched"
# ----------------------------------------------------------

NB, SEED = 2000, 42
RNG = np.random.default_rng(SEED)
SRC = {"A->B": "A", "B->A": "B"}      # source = the site trained on
TGT = {"A->B": "B", "B->A": "A"}      # target = the site tested on  (the reference site)
MODELS = ["RF", "xgboost", "lightgbm"]
DIRS   = ["A->B", "B->A"]

cross  = pd.read_csv(CROSS_FILE)
within = pd.read_csv(WITHIN_FILE)
_log = []
def out(s=""):
    print(s); _log.append(str(s))

def cross_auprc(direction, model, cond, arm):
    s = cross[(cross.direction==direction)&(cross.model==model)&
              (cross.condition==cond)&(cross.arm==arm)]
    return average_precision_score(s.y_true.values, s.prob.values)

def within_mean(site, model, cond, arm):
    w = within[(within.site==site)&(within.model==model)&
               (within.condition==cond)&(within.arm==arm)]
    return w.auprc.mean()

def boot_contrast(direction, model, cellA, cellB):
    """Paired target-set bootstrap of AUPRC(cellA) - AUPRC(cellB) on the SAME patients."""
    def get(cond, arm):
        s = cross[(cross.direction==direction)&(cross.model==model)&
                  (cross.condition==cond)&(cross.arm==arm)]
        return s.set_index("pid")[["y_true","prob"]]
    j = get(*cellA).join(get(*cellB), lsuffix="_a", rsuffix="_b")
    y, pa, pb = j.y_true_a.values, j.prob_a.values, j.prob_b.values
    obs = average_precision_score(y, pa) - average_precision_score(y, pb)
    n = len(y); d = np.empty(NB)
    for i in range(NB):
        ix = RNG.integers(0, n, n)
        if y[ix].sum() == 0:
            d[i] = np.nan; continue
        d[i] = average_precision_score(y[ix], pa[ix]) - average_precision_score(y[ix], pb[ix])
    return obs, np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)

# ---- target-site prevalence (drives the AUPRC scale; report it up front) ----
out("="*104)
out(f"PS-A ANALYSIS  [{OUT_PREFIX}]   target-site prevalence (AUPRC is prevalence-dependent)")
out("="*104)
for d in DIRS:
    u = cross[cross.direction==d].drop_duplicates("pid")
    out(f"  {d}: target = site {TGT[d]}   n={u.pid.nunique():5d}   prevalence={u.y_true.mean()*100:5.2f}%")

# =================== Q1  ORDERING RELIANCE (no correction needed) ===================
out("\n"+"="*104)
out("Q1  ORDERING RELIANCE, cross-site.   delta = AUPRC(full) - AUPRC(ordering_dropped)")
out("    Both arms scored on the SAME target patients -> prevalence cancels. Paired bootstrap.")
out("    <0 with CI excluding 0  =>  dropping ordering IMPROVES transfer (a shortcut).")
out("="*104)
q1 = []
for cond in ["1_none", "5_SMOTENC"]:
    for direction in DIRS:
        for model in MODELS:
            obs, lo, hi = boot_contrast(direction, model, (cond,"full"), (cond,"lab_indicators_only"))
            q1.append(dict(cond=cond, direction=direction, model=model,
                           delta=obs, lo=lo, hi=hi, sig=(lo>0 or hi<0)))
q1 = pd.DataFrame(q1)
out(q1.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
out(f"\n  mean ordering delta = {q1.delta.mean():+.4f} | CI excludes 0 in {int(q1.sig.sum())}/{len(q1)} cells")

# =================== Q2  RESAMPLING x TRANSPORT (target-referenced) ===================
out("\n"+"="*104)
out("Q2  RESAMPLING x TRANSPORT.  penalty = AUPRC(1_none) - AUPRC(5_SMOTENC)   (>0 = SMOTENC hurts)")
out("    within_tgt = train AND test on the TARGET site (CV).  cross = train on source, test on target.")
out("    Both are scored on the target distribution -> same prevalence -> DiD is clean.")
out("    DiD > 0  =>  SMOTENC hurts MORE when transferring than when staying in-site.  arm=full.")
out("="*104)
q2 = []
for direction in DIRS:
    tgt = TGT[direction]
    for model in MODELS:
        cx_none  = cross_auprc(direction, model, "1_none",    "full")
        cx_smote = cross_auprc(direction, model, "5_SMOTENC", "full")
        wt_none  = within_mean(tgt, model, "1_none",    "full")
        wt_smote = within_mean(tgt, model, "5_SMOTENC", "full")
        cross_pen  = cx_none - cx_smote
        within_pen = wt_none - wt_smote
        _, lo, hi = boot_contrast(direction, model, ("1_none","full"), ("5_SMOTENC","full"))
        q2.append(dict(
            direction=direction, model=model,
            within_tgt_pen=within_pen, cross_pen=cross_pen, DiD=cross_pen-within_pen,
            within_pen_rel=100*within_pen/wt_none, cross_pen_rel=100*cross_pen/cx_none,
            DiD_rel=100*cross_pen/cx_none - 100*within_pen/wt_none,
            cross_pen_lo=lo, cross_pen_hi=hi, cross_sig=(lo > 0)))
q2 = pd.DataFrame(q2)
out("--- absolute AUPRC penalties ---")
out(q2[["direction","model","within_tgt_pen","cross_pen","DiD","cross_pen_lo","cross_pen_hi","cross_sig"]]
    .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
out("\n--- RELATIVE penalties (% of that setting's own baseline; scale-free, use these) ---")
out(q2[["direction","model","within_pen_rel","cross_pen_rel","DiD_rel"]]
    .to_string(index=False, float_format=lambda v: f"{v:+.1f}"))
out(f"\n  mean within-target SMOTENC penalty : {q2.within_tgt_pen.mean():+.4f}  ({q2.within_pen_rel.mean():+.1f}%)")
out(f"  mean cross-site  SMOTENC penalty   : {q2.cross_pen.mean():+.4f}  ({q2.cross_pen_rel.mean():+.1f}%)")
out(f"  mean DiD (absolute)                : {q2.DiD.mean():+.4f}")
out(f"  mean DiD (relative)                : {q2.DiD_rel.mean():+.1f}%")
out(f"  cross-site penalty CI excludes 0 in: {int(q2.cross_sig.sum())}/{len(q2)} cells")

# =================== CONTEXT  transportability gap (oracle-referenced) ===================
out("\n"+"="*104)
out("CONTEXT  transportability gap = within_TARGET (oracle: trained on target) - cross (trained on source)")
out("         Both tested on the target site -> same prevalence -> a clean domain-shift measure.")
out("         arm=full, condition=1_none.")
out("="*104)
ctx = []
for direction in DIRS:
    tgt = TGT[direction]
    for model in MODELS:
        oracle = within_mean(tgt, model, "1_none", "full")
        cx     = cross_auprc(direction, model, "1_none", "full")
        src_ref = within_mean(SRC[direction], model, "1_none", "full")   # old (confounded) reference
        ctx.append(dict(direction=direction, model=model,
                        oracle_within_tgt=oracle, cross=cx,
                        gap=oracle-cx, pct_drop=100*(oracle-cx)/oracle,
                        source_within_ref=src_ref))
ctx = pd.DataFrame(ctx)
out(ctx.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
out(f"\n  mean oracle (train on target) = {ctx.oracle_within_tgt.mean():.3f}"
    f"  ->  cross (train on source) = {ctx.cross.mean():.3f}"
    f"   (mean relative drop {ctx.pct_drop.mean():.0f}%)")

q1.to_csv(f"{OUT_PREFIX}_Q1_ordering.csv", index=False)
q2.to_csv(f"{OUT_PREFIX}_Q2_resampling.csv", index=False)
ctx.to_csv(f"{OUT_PREFIX}_context.csv", index=False)
open(f"{OUT_PREFIX}_report.txt", "w").write("\n".join(_log))
out(f"\nsaved: {OUT_PREFIX}_Q1_ordering.csv, {OUT_PREFIX}_Q2_resampling.csv, "
    f"{OUT_PREFIX}_context.csv, {OUT_PREFIX}_report.txt")
