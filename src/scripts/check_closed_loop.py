import sys
from pathlib import Path

import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from analyse_episodes import load_world_model, load_episodes, check_threshold, RESULTS_PATH, TAU, LATE_WINDOW
from train_world_model import margin_labels, margin_constraints

H = 10 # rollout steps checked (the planning horizon)
STEPS = (1, 2, 5, 10) # rollout steps reported
NEAR = 0.3 # rows with a real margin below this (times the constraint's scale) count as near the edge
VEL_COLS = [3, 4] # velocimeter x, y in the raw observation

@nnx.jit
def _encode(world_model, obs):
    return world_model.encoder(obs) # the planner encodes with the online encoder

@nnx.jit
def _step(world_model, z, u):
    return world_model.dynamics(z, u)

@nnx.jit
def _margin(world_model, z):
    return world_model.margin_fn(z)

def pad(x, n):
    # Every call sees n rows, so each network compiles once rather than once per episode length
    return np.concatenate([x, np.zeros((n - len(x),) + x.shape[1:], x.dtype)])

def rollout_margin_errors(world_model, ep, constraints, n):
    # From every visited state, roll the model out with the actions the planner then executed and compare the margin
    # head with the real margins reached. err[j][t] is the error j + 1 steps after visiting step t, shape (T - j, K)
    T = len(ep["actions"])
    m_real = margin_labels(ep["next_obs"], constraints) # margin after each step, as the head was trained
    z = _encode(world_model, jnp.asarray(pad(ep["obs"], n)))
    err = []
    for j in range(min(H, T)):
        z = _step(world_model, z, jnp.asarray(pad(ep["actions"][j:], n))) # row t takes action t + j
        err.append(np.asarray(_margin(world_model, z))[:T - j] - m_real[j:])
    return err, m_real

def summary(errs, reals, c, scale):
    # Bias / RMS error at each reported step, on all rows and near the edge
    out = []
    for k in STEPS:
        if not errs.get(k):
            continue
        d, r = np.concatenate(errs[k])[:, c], np.concatenate(reals[k])[:, c]
        near = r < NEAR * scale
        out.append(f"step {k} {d.mean():+.3f}/{np.sqrt(np.mean(d ** 2)):.3f}"
                   + (f" ({d[near].mean():+.3f}/{np.sqrt(np.mean(d[near] ** 2)):.3f})" if near.any() else ""))
    return ", ".join(out)

def check(experiment, tags, controllers):
    exp_dir = RESULTS_PATH / experiment
    configs = [exp_dir / tag / f"{c}_config.yaml" for tag in tags for c in controllers]
    cfg = yaml.safe_load(open(next(p for p in configs if p.exists())))
    world_model, _ = load_world_model(cfg)
    constraints = margin_constraints(cfg)
    n = cfg["sim_max_steps"]
    print(f"\n{experiment}: margin head on the planner's own rollouts, bias/RMS error (near the edge, real margin < "
          f"{NEAR} scale) at steps {STEPS}; then, over the violated episodes, how long before the violation the plans "
          f"went over the threshold for good, and the real margin and speed at that point")
    for tag in tags:
        print(f"{tag}:")
        for controller in controllers:
            episodes = load_episodes(exp_dir / tag, controller)
            if not episodes:
                continue
            errs, reals, late_errs, late_reals = {}, {}, {}, {}
            leads, lead_margins, lead_speeds = [], [], []
            for _, ep in episodes:
                T = len(ep["actions"])
                err, m_real = rollout_margin_errors(world_model, ep, constraints, n)
                violated = bool(ep["violations"][-1])
                for j, e in enumerate(err):
                    errs.setdefault(j + 1, []).append(e)
                    reals.setdefault(j + 1, []).append(m_real[j:])
                    if violated: # visited steps in the last LATE_WINDOW steps before the violation
                        late = np.arange(T - j) >= T - LATE_WINDOW
                        late_errs.setdefault(j + 1, []).append(e[late])
                        late_reals.setdefault(j + 1, []).append(m_real[j:][late])
                if violated:
                    result, _ = check_threshold(ep, TAU)
                    run = result["final_over_run"]
                    leads.append(run)
                    if run > 0:
                        start = T - run # first step of the final run of over-threshold plans
                        lead_margins.append(margin_labels(ep["obs"][start:start + 1], constraints)[0, 0])
                        lead_speeds.append(np.linalg.norm(ep["obs"][start, VEL_COLS]))
            for c, con in enumerate(constraints):
                print(f"  {controller:5s} margin {c} ({con['label']}), every visited step: "
                      f"{summary(errs, reals, c, con['scale'])}")
                if leads:
                    print(f"  {'':5s} margin {c}, last {LATE_WINDOW} steps before a violation: "
                          f"{summary(late_errs, late_reals, c, con['scale'])}")
            if leads:
                warned = [l for l in leads if l > 0]
                print(f"  {'':5s} {len(leads)} violated: plans over the threshold for the last "
                      f"{np.median(leads):.0f} steps (median; {len(leads) - len(warned)} with none)"
                      + (f", starting at a real margin of {np.median(lead_margins):.2f} m and a speed of "
                         f"{np.median(lead_speeds):.2f} m/s (medians)" if warned else ""))

if __name__ == "__main__":
    EXPERIMENT = "M1_pointgoal1_c_1_5_seed_0"
    TAGS = [f"e1/eps_{eps}" for eps in [0.0, 0.1, 0.2, 0.3, 0.5]]
    CONTROLLERS = ["SCP", "MPPI", "CEM"]
    check(EXPERIMENT, TAGS, CONTROLLERS)