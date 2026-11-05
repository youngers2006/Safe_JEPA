import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

# Plan time per step, measured with nothing else running on the GPU slice (runs in parallel slow each other down, so
# their logged plan times are not used for the compute comparison): a few short episodes per planner, saved under
# timing/ in the world model's experiment folder, where analyse_results.py finds them. Run it on an idle MIG slice,
# e.g. CUDA_VISIBLE_DEVICES=<idle slice UUID> XLA_PYTHON_CLIENT_PREALLOCATE=false python check_timing.py
# It ends with the plan time statistics per step; python check_timing.py summary prints them for runs already done
WM_CFG = "configs/wmTraining/pg1_c_1_5_model.yaml"
COMMON = "configs/Planner/common.yaml"
ENV = "SafetyPointGoal1-v0"
SEEDS = [0, 1, 2]
STEPS = 200
BUDGETS = [64, 256, 512, 2048, 8192, 32768, 131072] # E2's
PLANNERS = [("configs/Planner/scp.yaml", {"seed_sampler": "configs/Planner/mppi.yaml"}, "scp_mppi"),
            ("configs/Planner/scp.yaml", {}, "scp"),
            *[(f"configs/Planner/{s}.yaml", {"num_seqs": J, "top_k": max(2, round(64 / 512 * J))}, f"{s}_J_{J}")
              for J in BUDGETS for s in ("mppi", "cem")]]

def summarise():
    # Median, mean, 90th and 99th percentile of the plan time per step over every timed step of each planner. A
    # sampler's time is nearly constant while SCP+MPPI's varies with its iterations, so the matched budget is the
    # smallest whose mean is at least SCP+MPPI's mean (at least as much compute per episode), with the share of
    # SCP+MPPI's steps that still took longer
    base = ROOT / "src/save_data/experiments" / yaml.safe_load(open(ROOT / WM_CFG))["data_dir"] / "timing"
    times = {}
    for _, _, name in PLANNERS:
        files = sorted((base / name).glob("*_seed_*.npz"))
        if files:
            times[name] = np.concatenate([np.load(f)["plan_times"] for f in files])
    print(f"\nPlan time per step (s), over every timed step in {base}:")
    print(f"  {'planner':16s} {'steps':>6s} {'median':>7s} {'mean':>7s} {'p90':>7s} {'p99':>7s}")
    for name, t in times.items():
        print(f"  {name:16s} {len(t):6d} {np.median(t):7.3f} {t.mean():7.3f} {np.percentile(t, 90):7.3f} "
              f"{np.percentile(t, 99):7.3f}")
    if "scp_mppi" not in times:
        return
    ref = times["scp_mppi"]
    for s in ("mppi", "cem"):
        matched = [J for J in BUDGETS if f"{s}_J_{J}" in times and times[f"{s}_J_{J}"].mean() >= ref.mean()]
        if not matched:
            if any(f"{s}_J_{J}" in times for J in BUDGETS):
                print(f"  {s.upper()}: no budget reaches SCP+MPPI's mean ({ref.mean():.3f} s); add larger budgets")
            continue
        t = times[f"{s}_J_{matched[0]}"]
        print(f"  {s.upper()}: matched budget {matched[0]} (mean {t.mean():.3f} s, at least SCP+MPPI's "
              f"{ref.mean():.3f} s); SCP+MPPI planned for longer than that on {np.mean(ref > t.mean()):.1%} of steps")

if __name__ == "__main__":
    if "summary" not in sys.argv[1:]:
        for planner, overrides, name in PLANNERS:
            run_point(SEEDS, WM_CFG, COMMON, planner, ENV,
                      {"margin_eps": 0.2, "sim_max_steps": STEPS, **overrides}, f"timing/{name}")
    summarise()
