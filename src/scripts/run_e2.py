import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

WM_CFGS = ["configs/wmTraining/pg1_c_1_5_model.yaml",
           "configs/wmTraining/pg1_c_inf_model.yaml"]
COMMON = "configs/Planner/common.yaml"
SAMPLERS = ["configs/Planner/cem.yaml", "configs/Planner/mppi.yaml"]
ENV = "SafetyPointGoal1-v0"
SEEDS = list(range(30))
THRESHOLD = 0.1
BUDGETS = [64, 256, 512, 2048, 8192, 32768, 131072]
ELITE_FRAC = 64 / 512 # held fixed, so top_k scales with the budget

def main():
    for wm_cfg in WM_CFGS:
        for J in tqdm(BUDGETS, leave=True):
            for planner in tqdm(SAMPLERS, leave=False):
                run_point(SEEDS, wm_cfg, COMMON, planner, ENV,
                        {"safety_threshold": THRESHOLD,
                        "num_seqs": J,
                        "top_k": max(2, round(ELITE_FRAC * J))},
                        f"e2/J_{J}")

if __name__ == "__main__":
    main()