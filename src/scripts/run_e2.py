import re
import sys
from pathlib import Path
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

COMMON = "configs/Planner/common.yaml"
SAMPLERS = {"MPPI": "configs/Planner/mppi.yaml", "CEM": "configs/Planner/cem.yaml"}
# Command line: the world model (a config name in configs/wmTraining), margin_eps values, budgets as J<sequences>
# and sampler names, so the budgets can run on separate GPU slices, e.g. python run_e2.py pg1_c_1_5_model 0.5 J131072.
# Without them: pg1_c_1_5_model, eps 0.2, every budget, MPPI only (the sampler SCP+MPPI starts from). The model's
# name gives its task, and its training windows its planning horizon
ARGS = sys.argv[1:]
MODEL = next((a for a in ARGS if (ROOT / f"configs/wmTraining/{a}.yaml").exists()), "pg1_c_1_5_model")
WM_CFG = f"configs/wmTraining/{MODEL}.yaml"
ENV = {"pg1": "SafetyPointGoal1-v0", "pg2": "SafetyPointGoal2-v0"}[MODEL[:3]]
HORIZON = yaml.safe_load(open(ROOT / WM_CFG))["train_horizon"] - 1
SEEDS = list(range(20))
BUDGETS = [int(a[1:]) for a in ARGS if re.fullmatch(r"J\d+", a)] or [64, 256, 512, 2048, 8192, 32768, 131072]
ELITE_FRAC = 64 / 512 # held fixed, so top_k scales with the budget
MARGIN_EPS = [float(a) for a in ARGS if re.fullmatch(r"-?\d*\.?\d+", a)] or [0.2]
RUN = [a for a in ARGS if a in SAMPLERS] or ["MPPI"]

def tag(eps, J):
    # eps 0.2 keeps E2's original folders (e2/J_<budget>); other eps get their own (e2/eps_<eps>/J_<budget>), so a new
    # eps never overwrites the existing results
    return f"e2/J_{J}" if eps == 0.2 else f"e2/eps_{eps:g}/J_{J}"

def main():
    for eps in MARGIN_EPS:
        for J in tqdm(BUDGETS, leave=True):
            for name in tqdm(RUN, leave=False):
                run_point(SEEDS, WM_CFG, COMMON, SAMPLERS[name], ENV,
                          {"num_seqs": J, "top_k": max(2, round(ELITE_FRAC * J)), "margin_eps": eps,
                           "horizon": HORIZON},
                          tag(eps, J))

if __name__ == "__main__":
    main()