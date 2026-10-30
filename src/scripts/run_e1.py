import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import run_point

COMMON = "configs/Planner/common.yaml"
PLANNERS = {"SCP+MPPI": ("configs/Planner/scp.yaml", {"seed_sampler": "configs/Planner/mppi.yaml"}),
            "MPPI": ("configs/Planner/mppi.yaml", {}),
            "CEM": ("configs/Planner/cem.yaml", {}),
            "SCP": ("configs/Planner/scp.yaml", {})} # pure SCP, the ablation
TASKS = {"pg1": ("configs/wmTraining/pg1_c_1_5_model.yaml", "SafetyPointGoal1-v0"),
         "pg2": ("configs/wmTraining/pg2_c_1_5_model.yaml", "SafetyPointGoal2-v0")}
# Command line: the task, margin_eps values, planner names and a seed range, so runs can be split across machines,
# e.g. python run_e1.py pg2 0.2 SCP 10:20. Without a task, pg1; without values, every margin_eps (the distance kept
# from every hazard's edge); without names, all but pure SCP; without a range, seeds 0 to 19
ARGS = sys.argv[1:]
WM_CFG, ENV = TASKS[next((a for a in ARGS if a in TASKS), "pg1")]
SEEDS = next((list(range(*map(int, a.split(":")))) for a in ARGS if ":" in a), list(range(20)))
MARGIN_EPS = ([float(a) for a in ARGS if a not in PLANNERS and a not in TASKS and ":" not in a]
              or [0.1, 0.2, 0.3, 0.5, 0.0])
RUN = [a for a in ARGS if a in PLANNERS] or ["SCP+MPPI", "MPPI", "CEM"]

def main():
    for eps in tqdm(MARGIN_EPS, leave=True):
        for planner, overrides in tqdm([PLANNERS[name] for name in RUN], leave=False):
            run_point(SEEDS, WM_CFG, COMMON, planner, ENV,
                      {"margin_eps": eps, **overrides}, f"e1/eps_{eps}")

if __name__ == "__main__":
    main()