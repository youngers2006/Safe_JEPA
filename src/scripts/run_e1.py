import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

WM_CFGS = ["configs/wmTraining/pg1_c_1_5_model.yaml",
           "configs/wmTraining/pg1_c_inf_model.yaml"]
COMMON = "configs/Planner/common.yaml"
PLANNERS = ["configs/Planner/cem.yaml",
            "configs/Planner/mppi.yaml",
            "configs/Planner/scp.yaml"]
ENV = "SafetyPointGoal1-v0"
SEEDS = list(range(30))
THRESHOLDS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]

def main():
    for wm_cfg in WM_CFGS:
        for p in tqdm(THRESHOLDS, leave=True):
            for planner in tqdm(PLANNERS, leave=False):
                run_point(SEEDS, wm_cfg, COMMON, planner, ENV,
                          {"safety_threshold": p}, f"e1/p_{p}")

if __name__ == "__main__":
    main()