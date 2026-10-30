import sys
from pathlib import Path
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

COMMON = "configs/Planner/common.yaml"
SAMPLERS = ["configs/Planner/mppi.yaml"] # the sampler SCP+MPPI starts from
# The world model on the command line (a config name in configs/wmTraining), e.g. python run_e2.py pg2_c_1_5_model;
# without it, pg1_c_1_5_model. Its name gives its task, and its training windows its planning horizon
MODEL = next((a for a in sys.argv[1:] if (ROOT / f"configs/wmTraining/{a}.yaml").exists()), "pg1_c_1_5_model")
WM_CFG = f"configs/wmTraining/{MODEL}.yaml"
ENV = {"pg1": "SafetyPointGoal1-v0", "pg2": "SafetyPointGoal2-v0"}[MODEL[:3]]
HORIZON = yaml.safe_load(open(ROOT / WM_CFG))["train_horizon"] - 1
SEEDS = list(range(20))
BUDGETS = [64, 256, 512, 2048, 8192, 32768, 131072]
ELITE_FRAC = 64 / 512 # held fixed, so top_k scales with the budget
MARGIN_EPS = 0.2 # an E1 point where the planners differ, so SCP+MPPI's E1 result sits on the same axes

def main():
    for J in tqdm(BUDGETS, leave=True):
        for planner in tqdm(SAMPLERS, leave=False):
            run_point(SEEDS, WM_CFG, COMMON, planner, ENV,
                      {"num_seqs": J, "top_k": max(2, round(ELITE_FRAC * J)), "margin_eps": MARGIN_EPS,
                       "horizon": HORIZON},
                      f"e2/J_{J}")

if __name__ == "__main__":
    main()