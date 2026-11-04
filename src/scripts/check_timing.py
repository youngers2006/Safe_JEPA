import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

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

if __name__ == "__main__":
    for planner, overrides, name in PLANNERS:
        rows = run_point(SEEDS, WM_CFG, COMMON, planner, ENV,
                         {"margin_eps": 0.2, "sim_max_steps": STEPS, **overrides}, f"timing/{name}")
        print(f"{name}: median plan time {sorted(r['plan_time_med'] for r in rows)[len(rows) // 2]:.3f} s")