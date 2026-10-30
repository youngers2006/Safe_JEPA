import sys
from pathlib import Path
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

COMMON = "configs/Planner/common.yaml"
PLANNERS = {"SCP+MPPI": ("configs/Planner/scp.yaml", {"seed_sampler": "configs/Planner/mppi.yaml"}),
            "MPPI": ("configs/Planner/mppi.yaml", {}),
            "CEM": ("configs/Planner/cem.yaml", {}),
            "SCP": ("configs/Planner/scp.yaml", {})} # pure SCP, the ablation
# Command line: the world model (a config name in configs/wmTraining), margin_eps values, planner names and a seed
# range, so runs can be split across processes, e.g. python run_e1.py pg2_c_1_2_model 0.2 SCP 0:30. Without a model,
# pg1_c_1_5_model; without values, every margin_eps (the distance kept from every hazard's edge); without names, all
# but pure SCP; without a range, seeds 0 to 19. The model's name gives its task, and its training windows its planning
# horizon (train_horizon - 1)
ARGS = sys.argv[1:]
MODEL = next((a for a in ARGS if (ROOT / f"configs/wmTraining/{a}.yaml").exists()), "pg1_c_1_5_model")
WM_CFG = f"configs/wmTraining/{MODEL}.yaml"
ENV = {"pg1": "SafetyPointGoal1-v0", "pg2": "SafetyPointGoal2-v0"}[MODEL[:3]]
HORIZON = yaml.safe_load(open(ROOT / WM_CFG))["train_horizon"] - 1
SEEDS = next((list(range(*map(int, a.split(":")))) for a in ARGS if ":" in a), list(range(20)))
MARGIN_EPS = ([float(a) for a in ARGS if a not in PLANNERS and a != MODEL and ":" not in a]
              or [0.1, 0.2, 0.3, 0.5, 0.0])
RUN = [a for a in ARGS if a in PLANNERS] or ["SCP+MPPI", "MPPI", "CEM"]

def main():
    for eps in tqdm(MARGIN_EPS, leave=True):
        for planner, overrides in tqdm([PLANNERS[name] for name in RUN], leave=False):
            run_point(SEEDS, WM_CFG, COMMON, planner, ENV,
                      {"margin_eps": eps, "horizon": HORIZON, **overrides}, f"e1/eps_{eps}")

if __name__ == "__main__":
    main()
