import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

WM_CFGS = ["configs/wmTraining/pg1_c_1_5_model.yaml"]
COMMON = "configs/Planner/common.yaml"
PLANNERS = {"SCP+MPPI": ("configs/Planner/scp.yaml", {"seed_sampler": "configs/Planner/mppi.yaml"}),
            "MPPI": ("configs/Planner/mppi.yaml", {}),
            "CEM": ("configs/Planner/cem.yaml", {}),
            "SCP": ("configs/Planner/scp.yaml", {})} # pure SCP, the ablation
ENV = "SafetyPointGoal1-v0"
SEEDS = list(range(30))
# Command line: margin_eps values and planner names, so each runs in its own process, e.g. python run_e1.py 0.2 SCP.
# Without values, every margin_eps (the distance kept from every hazard's edge); without names, all but pure SCP
ARGS = sys.argv[1:]
MARGIN_EPS = [float(a) for a in ARGS if a not in PLANNERS] or [0.0, 0.1, 0.2, 0.3]
RUN = [a for a in ARGS if a in PLANNERS] or ["SCP+MPPI", "MPPI", "CEM"]

def main():
    for wm_cfg in WM_CFGS:
        for eps in tqdm(MARGIN_EPS, leave=True):
            for planner, overrides in tqdm([PLANNERS[name] for name in RUN], leave=False):
                run_point(SEEDS, wm_cfg, COMMON, planner, ENV,
                          {"margin_eps": eps, **overrides}, f"e1/eps_{eps}")

if __name__ == "__main__":
    main()