import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

WM_CFGS = ["configs/wmTraining/pg1_c_1_5_model.yaml"]
COMMON = "configs/Planner/common.yaml"
PLANNERS = ["configs/Planner/scp.yaml",
            "configs/Planner/mppi.yaml",
            "configs/Planner/cem.yaml"]
ENV = "SafetyPointGoal1-v0"
SEEDS = list(range(30))
MARGIN_EPS = [0.0, 0.025, 0.05, 0.1, 0.2] # distance kept from every hazard's edge

def main():
    for wm_cfg in WM_CFGS:
        for eps in tqdm(MARGIN_EPS, leave=True):
            for planner in tqdm(PLANNERS, leave=False):
                run_point(SEEDS, wm_cfg, COMMON, planner, ENV,
                          {"safety_threshold": p}, f"e1/p_{eps}")

if __name__ == "__main__":
    main()