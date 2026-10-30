import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

WM_CFGS = ["configs/wmTraining/pg1_c_1_5_model.yaml"]
COMMON = "configs/Planner/common.yaml"
SAMPLERS = ["configs/Planner/mppi.yaml"] # the sampler SCP+MPPI starts from
ENV = "SafetyPointGoal1-v0"
SEEDS = list(range(30))
BUDGETS = [64, 256, 512, 2048, 8192, 32768, 131072]
ELITE_FRAC = 64 / 512 # held fixed, so top_k scales with the budget
MARGIN_EPS = 0.2 # an E1 point where the planners differ, so SCP+MPPI's E1 result sits on the same axes

def main():
    for wm_cfg in WM_CFGS:
        for J in tqdm(BUDGETS, leave=True):
            for planner in tqdm(SAMPLERS, leave=False):
                run_point(SEEDS, wm_cfg, COMMON, planner, ENV,
                        {"num_seqs": J,
                        "top_k": max(2, round(ELITE_FRAC * J)),
                        "margin_eps": MARGIN_EPS},
                        f"e2/J_{J}")

if __name__ == "__main__":
    main()