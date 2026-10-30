import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

# Horizon test at eps 0.2: each planning horizon with the world model trained on windows one step longer. H 10 is E1's
# eps 0.2 point. Horizons on the command line, e.g. python run_e1_SCP_sampling_single.py 30; without them, all
WM_CFGS = {20: "configs/wmTraining/pg1_c_1_5_model_h20.yaml",
           30: "configs/wmTraining/pg1_c_1_5_model_h30.yaml"}
COMMON = "configs/Planner/common.yaml"
PLANNERS = {"SCP+MPPI": ("configs/Planner/scp.yaml", {"seed_sampler": "configs/Planner/mppi.yaml"})}
ENV = "SafetyPointGoal1-v0"
SEEDS = list(range(10))
MARGIN_EPS = 0.2
HORIZONS = [int(a) for a in sys.argv[1:]] or list(WM_CFGS)

def main():
    for H in tqdm(HORIZONS, leave=True):
        for planner, overrides in tqdm(PLANNERS.values(), leave=False):
            run_point(SEEDS, WM_CFGS[H], COMMON, planner, ENV,
                      {"margin_eps": MARGIN_EPS, "horizon": H, **overrides}, f"e4/H_{H}")

if __name__ == "__main__":
    main()