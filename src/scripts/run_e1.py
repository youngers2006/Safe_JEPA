import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

WM_CFGS = ["configs/wmTraining/pg1_c_1_5_model.yaml"]
COMMON = "configs/Planner/common.yaml"
PLANNERS = [("configs/Planner/scp.yaml", {"seed_sampler": "configs/Planner/mppi.yaml"}), # saved as SCP+MPPI
            ("configs/Planner/mppi.yaml", {}),
            ("configs/Planner/cem.yaml", {})]
ENV = "SafetyPointGoal1-v0"
SEEDS = list(range(30))
# Distance kept from every hazard's edge; one process per value runs them in parallel: python run_e1.py 0.2
MARGIN_EPS = [float(a) for a in sys.argv[1:]] or [0.0, 0.1, 0.2, 0.3]

def main():
    for wm_cfg in WM_CFGS:
        for eps in tqdm(MARGIN_EPS, leave=True):
            for planner, overrides in tqdm(PLANNERS, leave=False):
                run_point(SEEDS, wm_cfg, COMMON, planner, ENV,
                          {"margin_eps": eps, **overrides}, f"e1/eps_{eps}")

if __name__ == "__main__":
    main()