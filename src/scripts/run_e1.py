import os
import subprocess
import sys
import time
from pathlib import Path

# Must be set before anything imports JAX: several processes share one GPU
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]

# Fixed for the whole sweep. Every point shares this world model and critic,
# which is what makes the result a within-model trend.
WM_CFG   = "configs/wmTraining/regularity_constrained_model.yaml"
COMMON   = "configs/Planner/common.yaml"
PLANNERS = ["configs/Planner/cem.yaml",
            "configs/Planner/mppi.yaml",
            "configs/Planner/scp.yaml"]
ENV = "SafetyPointGoal1-v0"
SEEDS = list(range(100))
THRESHOLDS = [0.5, 0.2, 0.1, 0.05, 0.02]
MAX_PARALLEL = 5        # lower if the GPU runs out of memory; 1 = sequential
LOG_DIR = ROOT / "logs"

def run_one(p):
    # Imported here so the launcher process never initialises JAX
    sys.path.insert(0, str(ROOT / "src/scripts"))
    from .sweep_runner import run_point

    for planner in PLANNERS:
        run_point(seeds=SEEDS, wm_cfg=WM_CFG, common_cfg=COMMON,
                  planner_cfg=planner, env_name=ENV,
                  overrides={"safety_threshold": p}, tag=f"p_{p}")

def launch(thresholds):
    LOG_DIR.mkdir(exist_ok=True)
    pending, running, failed = list(thresholds), [], []

    try:
        while pending or running:
            while pending and len(running) < MAX_PARALLEL:
                p = pending.pop(0)
                log = open(LOG_DIR / f"e1_p{p}.log", "w")
                proc = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), str(p)],
                    stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                running.append((p, proc, log))
                print(f"started  p={p:<5} pid {proc.pid}  -> {log.name}", flush=True)

            time.sleep(5)

            still_running = []
            for p, proc, log in running:
                if proc.poll() is None:
                    still_running.append((p, proc, log))
                    continue
                log.close()
                if proc.returncode == 0:
                    print(f"finished p={p}", flush=True)
                else:
                    print(f"FAILED   p={p} (exit {proc.returncode}), see {log.name}", flush=True)
                    failed.append(p)
            running = still_running

    finally:
        # Ctrl-C or a launcher crash: don't leave orphans holding the GPU
        for _, proc, log in running:
            proc.terminate()
            log.close()

    if failed:
        sys.exit(f"failed thresholds: {failed}")
    print("all thresholds finished", flush=True)

if __name__ == "__main__":
    args = [float(a) for a in sys.argv[1:]]
    if len(args) == 1:
        run_one(args[0])                  # a child, or a single foreground run
    else:
        launch(args or THRESHOLDS)