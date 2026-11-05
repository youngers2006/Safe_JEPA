import csv
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

# Plan time per step, measured with nothing else running on the GPU slice (runs in parallel slow each other down, so
# their logged plan times are not used for the compute comparison): a few short episodes per planner, saved under
# timing/ in the world model's experiment folder, where analyse_results.py finds them, and summarised in
# timing/timing_summary.csv. Run it on an idle MIG slice,
# e.g. CUDA_VISIBLE_DEVICES=<idle slice UUID> XLA_PYTHON_CLIENT_PREALLOCATE=false python check_timing.py
# Command line: J<sequences> times only those sampler budgets (SCP's timing is reused), e.g. python check_timing.py
# J35392; summary only rewrites the CSV from the timing runs already done
WM_CFG = "configs/wmTraining/pg1_c_1_5_model.yaml"
COMMON = "configs/Planner/common.yaml"
ENV = "SafetyPointGoal1-v0"
SEEDS = [0, 1, 2]
STEPS = 200
BUDGETS = [64, 256, 512, 2048, 8192, 32768, 131072] # E2's
SAMPLERS = ("mppi", "cem")
ARGS = sys.argv[1:]
ONLY = [int(a[1:]) for a in ARGS if re.fullmatch(r"J\d+", a)]
PLANNERS = ([] if ONLY else [("configs/Planner/scp.yaml", {"seed_sampler": "configs/Planner/mppi.yaml"}, "scp_mppi"),
                             ("configs/Planner/scp.yaml", {}, "scp")]) \
    + [(f"configs/Planner/{s}.yaml", {"num_seqs": J, "top_k": max(2, round(64 / 512 * J))}, f"{s}_J_{J}")
       for J in (ONLY or BUDGETS) for s in SAMPLERS]

def summarise():
    # One row per timed planner: plan time per step over every timed step (median, mean, 90th and 99th percentile),
    # and for the samplers how they compare with SCP+MPPI. A sampler's time is nearly constant while SCP+MPPI's varies
    # with its iterations, so a budget is matched when its mean is at least SCP+MPPI's (at least as much compute per
    # episode). Where SCP+MPPI's mean falls between two budgets, a budget interpolated between them is suggested
    base = ROOT / "src/save_data/experiments" / yaml.safe_load(open(ROOT / WM_CFG))["data_dir"] / "timing"
    times = {}
    for d in sorted(base.iterdir()) if base.exists() else []:
        files = sorted(d.glob("*_seed_*.npz")) if d.is_dir() else []
        if files:
            times[d.name] = np.concatenate([np.load(f)["plan_times"] for f in files])
    ref = times.get("scp_mppi")
    rows = []
    for name, t in times.items():
        m = re.fullmatch(r"(mppi|cem)_J_(\d+)", name)
        row = {"name": name, "controller": {"scp_mppi": "SCP+MPPI", "scp": "SCP"}.get(name, m.group(1).upper() if m
                                                                                       else name),
               "sequences": int(m.group(2)) if m else "", "steps": len(t), "median_s": np.median(t),
               "mean_s": t.mean(), "p90_s": np.percentile(t, 90), "p99_s": np.percentile(t, 99)}
        if ref is not None:
            row["mean_over_scp_mppi"] = t.mean() / ref.mean()
            if m: # the share of SCP+MPPI's steps that took longer than this sampler's (nearly constant) time
                row["scp_mppi_steps_slower"] = np.mean(ref > t.mean())
        rows.append(row)
    if ref is not None:
        for s in SAMPLERS:
            budgets = sorted(((r["sequences"], r) for r in rows if r["name"].startswith(f"{s}_J_")),
                             key=lambda jr: jr[0])
            above = [(J, r) for J, r in budgets if r["mean_s"] >= ref.mean()]
            below = [(J, r) for J, r in budgets if r["mean_s"] < ref.mean()]
            if above:
                above[0][1]["matched"] = "yes: the smallest budget whose mean is at least SCP+MPPI's"
            if above and below:
                (J0, r0), (J1, r1) = below[-1], above[0]
                J = J0 + (ref.mean() - r0["mean_s"]) / (r1["mean_s"] - r0["mean_s"]) * (J1 - J0)
                J = int(np.ceil(J / 64) * 64)
                if J0 < J < J1:
                    rows.append({"name": f"{s}_J_{J}", "controller": s.upper(), "sequences": J,
                                 "matched": f"suggested: interpolated between {J0} and {J1} for SCP+MPPI's mean of "
                                            f"{ref.mean():.3f} s; time it with: python check_timing.py J{J}"})
    rank = {"SCP+MPPI": 0, "SCP": 1, "MPPI": 2, "CEM": 3} # each sampler by budget, a suggestion among them
    rows.sort(key=lambda r: (rank.get(r["controller"], len(rank)), r["sequences"] or 0))
    path = base / "timing_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print(f"Plan time statistics written to {path}")

if __name__ == "__main__":
    if "summary" not in ARGS:
        for planner, overrides, name in PLANNERS:
            run_point(SEEDS, WM_CFG, COMMON, planner, ENV,
                      {"margin_eps": 0.2, "sim_max_steps": STEPS, **overrides}, f"timing/{name}")
    summarise()
