import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

# Time-matched samplers for the horizon test at eps 0.2: MPPI and CEM at the budget whose idle plan time per step
# roughly matches SCP+MPPI's at that horizon (check_timing.py). Saved in their own folder, since run_point names files
# by planner only; analyse_results.py folds them into e4/H_<H> as e.g. "MPPI@131072". To split the runs over GPU
# slices, edit HORIZONS, PLANNERS and SEEDS for each launch (seed ranges must not overlap)
WM_CFGS = {20: "configs/wmTraining/pg1_c_1_5_model_h20.yaml",
           30: "configs/wmTraining/pg1_c_1_5_model_h30.yaml"}
BUDGETS = {20: 95040, 30: 131072} # sequences per iteration; top_k is an eighth of it, as at H10
COMMON = "configs/Planner/common.yaml"
PLANNERS = {"MPPI": "configs/Planner/mppi.yaml",
            "CEM": "configs/Planner/cem.yaml"}
ENV = "SafetyPointGoal1-v0"
SEEDS = list(range(30))
MARGIN_EPS = 0.2
HORIZONS = [20, 30]

def main():
    for H in tqdm(HORIZONS, leave=True):
        J = BUDGETS[H]
        for planner in tqdm(PLANNERS.values(), leave=False):
            run_point(SEEDS, WM_CFGS[H], COMMON, planner, ENV,
                      {"margin_eps": MARGIN_EPS, "horizon": H, "num_seqs": J, "top_k": J // 8},
                      f"e4_extended_budget_matched/H_{H}")

if __name__ == "__main__":
    main()
