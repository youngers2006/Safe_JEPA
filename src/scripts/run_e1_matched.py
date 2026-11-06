import re
import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

# Time-matched baselines: each sampler at J sequences per iteration (the budget whose mean plan time is at least
# SCP+MPPI's; see check_timing.py), and MPPI with the one-step safety filter, saved beside the 512-sequence runs under
# their own names. Command line: margin_eps values and planner names, so each can run on its own GPU slice, e.g.
# python run_e1_matched.py 0.2 MPPI+CBF@65536; without them, every margin_eps and every planner below
WM_CFGS = ["configs/wmTraining/pg1_c_1_5_model.yaml"]
COMMON = "configs/Planner/common.yaml"
ENV = "SafetyPointGoal1-v0"
SEEDS = list(range(30))
J = 65536
BUDGET = {"num_seqs": J, "top_k": J // 8} # top_k scales with the budget, as in E2
PLANNERS = {f"MPPI@{J}": ("configs/Planner/mppi.yaml", BUDGET),
            f"CEM@{J}": ("configs/Planner/cem.yaml", BUDGET),
            f"MPPI+CBF@{J}": ("configs/Planner/mppi.yaml", {**BUDGET, "safety_filter": True})}
ARGS = sys.argv[1:]
MARGIN_EPS = [float(a) for a in ARGS if re.fullmatch(r"-?\d*\.?\d+", a)] or [0.1, 0.2, 0.3, 0.5, 0.0]
RUN = [a for a in ARGS if a in PLANNERS] or list(PLANNERS)

def main():
    for wm_cfg in WM_CFGS:
        for eps in tqdm(MARGIN_EPS, leave=True):
            for name in tqdm(RUN, leave=False):
                planner, overrides = PLANNERS[name]
                run_point(SEEDS, wm_cfg, COMMON, planner, ENV,
                          {"margin_eps": eps, **overrides}, f"e1/eps_{eps}", name=name)

if __name__ == "__main__":
    main()
