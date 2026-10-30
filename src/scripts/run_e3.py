import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

# Lipschitz ablation: E1's eps 0.2 point with world models that differ from the main one only in lipschitz_bound. The
# main (c 1.5) model's arm is E1's own eps 0.2 point, so only the other models are run, with E1's seeds
WM_CFGS = ["configs/wmTraining/pg1_c_inf_model.yaml"]
COMMON = "configs/Planner/common.yaml"
PLANNERS = [("configs/Planner/scp.yaml", {"seed_sampler": "configs/Planner/mppi.yaml"}), # saved as SCP+MPPI
            ("configs/Planner/mppi.yaml", {})]
ENV = "SafetyPointGoal1-v0"
SEEDS = list(range(30))
MARGIN_EPS = 0.2

def main():
    for wm_cfg in WM_CFGS:
        for planner, overrides in tqdm(PLANNERS, leave=True):
            run_point(SEEDS, wm_cfg, COMMON, planner, ENV,
                      {"margin_eps": MARGIN_EPS, **overrides}, f"e1/eps_{MARGIN_EPS}")

if __name__ == "__main__":
    main()
