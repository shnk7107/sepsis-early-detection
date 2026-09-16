#!/usr/bin/env python3
"""
run_all.py — pipeline driver / reproducibility manifest.

WHAT THIS IS
    A single entry point that documents the canonical execution order of every
    stage, records which script version is authoritative where duplicates exist,
    and verifies that each stage's declared outputs are present.

WHAT THIS IS NOT
    Not a one-shot local re-run. Stages 01-08 model fitting was executed on
    Google Colab (GPU/long-running; see each stage's header for its Colab setup
    block). This driver reproduces the ORDER and the ANALYSIS steps, and audits
    outputs. Use --check to verify a delivered repo; use --list to print the
    order for manual re-execution.

USAGE
    python run_all.py --list     # print canonical stage order + authoritative scripts
    python run_all.py --check    # verify every declared output file exists
    python run_all.py --analysis # re-run the light analysis-only steps locally

SEED POLICY
    All stochastic operations use SEED = 42 (train/test split, StratifiedKFold,
    RandomForest/XGBoost/LightGBM, and every imblearn sampler). PS-A bootstrap
    uses NB = 2000 iterations with the same SEED = 42.
    Verified: `grep -rn "SEED" --include=*.py .` -> 42 everywhere.

RESAMPLING / LEAKAGE NOTE
    Resampling is a STEP INSIDE an imblearn Pipeline handed to cross_validate,
    so every sampler is fit on the training portion of each CV fold only. The
    held-out test PIDs are frozen in stage 03 before any fitting. Resampling
    therefore cannot leak across the split by construction.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# CANONICAL PIPELINE ORDER
#   script  = authoritative script for that stage (see NOTE where duplicates exist)
#   outputs = files the stage is expected to produce
#   venue   = "colab" (long model fitting) or "local" (analysis-only arithmetic)
# ---------------------------------------------------------------------------
STAGES = [
    dict(
        id="01", name="Data loading",
        script="01_data_loading/sepsis_loader.py",
        venue="colab",
        outputs=["01_data_loading/patient_index.csv"],
        note="Loads PhysioNet 2019 psv files -> patient index.",
    ),
    dict(
        id="02a", name="Windowing (6h pre-onset)",
        script="02_windowing_and_matching/Step_2_files/sepsis_window.py",
        venue="colab",
        outputs=["02_windowing_and_matching/Step_2_files/window_manifest_septic.csv"],
        note="Leakage-controlled window: strictly before first positive label.",
    ),
    dict(
        id="02b", name="Within-site position matching",
        script="02_windowing_and_matching/Step_3_files/sepsis_match.py",
        venue="colab",
        outputs=["02_windowing_and_matching/Step_3_files/window_manifest_control.csv"],
        note="Controls matched within site on window-end position.",
    ),
    dict(
        id="02c", name="Feature aggregation",
        script="02_windowing_and_matching/Step_4_file/sepsis_aggregate.py",
        venue="colab",
        outputs=["02_windowing_and_matching/Step_4_file/feature_tags.csv"],
        note="243 features; each source var -> value channel + indicator channel.",
    ),
    dict(
        id="03", name="Frozen train/test split",
        script="03_train_test_split/sepsis_day3_skeleton.py",
        venue="colab",
        outputs=["03_train_test_split/day3_test_pids.json"],
        note="Split frozen here. Every later stage asserts against these PIDs.",
    ),
    dict(
        id="04", name="Ablation grid (8 resampling x 3 arms)",
        script="04_ablation_grid/sepsis_day4_grid.py",
        venue="colab",
        outputs=["04_ablation_grid/day4_grid_results(output of sepsis_day4_grid).csv",
                 "04_ablation_grid/Analysis of day4 grid/2.Additional supporting materials/2.perfold/day4_perfold.csv"],
        note="Component-wise ablation. Collinearity (Creatinine<->BUN 0.97) "
             "motivates the panel-level arms in stage 05.",
    ),
    dict(
        id="05", name="Panel decomposition (+ SOFA-values arm)",
        script="05_panel_decomposition/sepsis_day5_panel.py",
        venue="colab",
        outputs=["05_panel_decomposition/output/day5_decomposition.csv"],
        note="Arms: lab_panel (all labs), lab_indicators_only (circularity), "
             "sofa_values_dropped (label-definition check).",
    ),
    dict(
        id="06", name="Repeated CV + Nadeau-Bengio significance",
        script="06_significance_testing/sepsis_day6_repeated.py",
        venue="colab",
        outputs=["06_significance_testing/Output/day6_significance.csv"],
        note="5x5 repeated CV. Produces the naive-vs-corrected contrast.",
    ),
    dict(
        id="07", name="Cross-model validation (RF/XGB/LGBM)",
        script="07_crossmodel_validation/sepsis_day7_models.py",
        venue="colab",
        outputs=["07_crossmodel_validation/day7_crossmodel_significance.csv"],
        note="Replicates the stage-06 collapse across three model families.",
    ),
    dict(
        id="08a", name="PS-A matched-prevalence cross-site run",
        script="08_crosssite_analysis/Mirror Arm Patch/Scripts/psA_matched_run.py",
        venue="colab",
        outputs=["08_crosssite_analysis/Mirror Arm Patch/Outputs/psA_withinsrc_perfold.csv"],
        note="AUTHORITATIVE VERSION = Mirror Arm Patch (6 arms). Supersedes "
             "1.Matched_Run/psA_matched_run.py (2 arms) and "
             "Enhanced_Matched/psA_matched_run.py (5 arms). The mirror version "
             "adds lab_values_dropped and produced the reconstruction numbers.",
    ),
    dict(
        id="08b", name="PS-A native-prevalence cross-site run",
        script="08_crosssite_analysis/Mirror Arm Patch/Scripts/psA_matched_run_native.py",
        venue="colab",
        outputs=["08_crosssite_analysis/Mirror Arm Patch/Outputs/psA_native_withinsrc_perfold.csv"],
        note="AUTHORITATIVE VERSION = Mirror Arm Patch (6 arms). Requires the "
             "native dataset built by 2.Native_Run/Stage 1/Input_files/native_build.py.",
    ),
    dict(
        id="08c", name="PS-A analysis — native inputs",
        script="08_crosssite_analysis/2.Native_Run/Stage 3/Pass 1/psA_prevalence_aware_analysis.py",
        venue="local",
        outputs=["08_crosssite_analysis/2.Native_Run/Stage 3/Pass 1/Outputs/psA_native_report.txt"],
        note="Pass 1 and Pass 2 are the SAME analysis pointed at different "
             "inputs, not versions. Pass 1 = native files. Both are required.",
    ),
    dict(
        id="08d", name="PS-A analysis — matched inputs",
        script="08_crosssite_analysis/2.Native_Run/Stage 3/Pass 2/psA_prevalence_aware_analysis.py",
        venue="local",
        outputs=["08_crosssite_analysis/2.Native_Run/Stage 3/Pass 2/Output/psA_matched_report.txt"],
        note="Pass 2 = matched files. See 08c.",
    ),
    dict(
        id="08e", name="PS-1 imputation diagnostic (MICE)",
        script="08_crosssite_analysis/Extra/psA_imputation_diagnostic_fixed.py",
        venue="colab",
        outputs=["08_crosssite_analysis/Extra/psA_imputation_diagnostic.csv"],
        note="Tests whether the value/ordering redundancy survives conditional "
             "imputation. Uses HistGradientBoostingRegressor (NaN-native).",
    ),
]

# Canonical corrected results (produced by post-hoc arithmetic on committed per-fold data)
RESULTS = [
    "results/RESULTS_primary_RF_1none.csv",
    "results/RESULTS_robustness_sweep.csv",
    "results/RESULTS_channel_reconstruction.csv",
]


def cmd_list():
    print("CANONICAL PIPELINE ORDER\n" + "=" * 78)
    for s in STAGES:
        print(f"\n[{s['id']}] {s['name']}   ({s['venue']})")
        print(f"     script : {s['script']}")
        for o in s["outputs"]:
            print(f"     output : {o}")
        if s.get("note"):
            for line in _wrap(s["note"], 68):
                print(f"     note   : {line}" if line is s["note"][:len(line)] else f"              {line}")
    print("\n" + "=" * 78)
    print("CANONICAL RESULTS (report these; per-stage files are provenance):")
    for r in RESULTS:
        print(f"     {r}")


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line); line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


def cmd_check():
    missing, present = [], 0
    print("OUTPUT AUDIT\n" + "=" * 78)
    for s in STAGES:
        for o in s["outputs"]:
            if (ROOT / o).exists():
                present += 1
            else:
                missing.append((s["id"], o))
    for r in RESULTS:
        if (ROOT / r).exists():
            present += 1
        else:
            missing.append(("results", r))
    print(f"present : {present}")
    print(f"missing : {len(missing)}")
    for sid, o in missing:
        print(f"  [{sid}] MISSING  {o}")
    return 1 if missing else 0


def cmd_analysis():
    """Re-run only the local analysis stages (no model fitting)."""
    rc = 0
    for s in STAGES:
        if s["venue"] != "local":
            continue
        script = ROOT / s["script"]
        if not script.exists():
            print(f"[{s['id']}] SKIP (script not found): {s['script']}")
            continue
        print(f"[{s['id']}] running {s['script']}")
        r = subprocess.run([sys.executable, str(script.name)], cwd=str(script.parent))
        rc |= r.returncode
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sepsis circularity pipeline driver")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="print canonical stage order")
    g.add_argument("--check", action="store_true", help="verify declared outputs exist")
    g.add_argument("--analysis", action="store_true", help="re-run local analysis stages")
    a = ap.parse_args()
    if a.list:
        cmd_list(); sys.exit(0)
    if a.check:
        sys.exit(cmd_check())
    if a.analysis:
        sys.exit(cmd_analysis())
