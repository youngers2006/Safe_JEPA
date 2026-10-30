import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

WM_CFGS = ["configs/wmTraining/pg1_c_1_5_model.yaml"]
COMMON = "configs/Planner/common.yaml"
PLANNERS = {"SCP+MPPI": ("configs/Planner/scp.yaml", {"seed_sampler": "configs/Planner/mppi.yaml"})}
ENV = "SafetyPointGoal1-v0"
SEEDS = list(range(30))
MARGIN_EPS = [0.1, 0.2, 0.3, 0.5] # distance kept from every hazard's edge

def main():
    for wm_cfg in WM_CFGS:
        for eps in tqdm(MARGIN_EPS, leave=True):
            for planner in tqdm(PLANNERS, leave=False):
                run_point(SEEDS, wm_cfg, COMMON, planner, ENV,
                          {"margin_eps": eps}, f"e1/eps_{eps}")

if __name__ == "__main__":
    main()