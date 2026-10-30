import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

# Plan time per step, measured with nothing else running on the machine (runs in parallel slow each other down, so
# their logged plan times are not used for the compute comparison). Also a quick check that every planner runs
WM_CFG = "configs/wmTraining/pg1_c_1_5_model.yaml"
COMMON = "configs/Planner/common.yaml"
ENV = "SafetyPointGoal1-v0"
STEPS = 200
BUDGETS = [512, 2048, 8192, 32768, 131072]
PLANNERS = [("configs/Planner/scp.yaml", {"seed_sampler": "configs/Planner/mppi.yaml"}, "scp_mppi"),
            *[("configs/Planner/mppi.yaml", {"num_seqs": J, "top_k": max(2, round(64 / 512 * J))}, f"mppi_J_{J}")
              for J in BUDGETS]]

if __name__ == "__main__":
    for planner, overrides, name in PLANNERS:
        rows = run_point([0], WM_CFG, COMMON, planner, ENV,
                         {"margin_eps": 0.2, "sim_max_steps": STEPS, **overrides}, f"timing/{name}")
        print(f"{name}: median plan time {rows[0]['plan_time_med']:.3f} s over {rows[0]['steps']} steps")
