import sys
import time
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import yaml
from jax.experimental import enable_x64

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import build_cfg, make_agent
from train_world_model import margin_labels
from safe_jepa.SCP_control.SCPSolver import SCPSolver
from safe_jepa.SCP_control.SCPSolverDirect import DirectSCPSolver

N_STATES = 100 # dataset states, half of them near a constraint's edge, each with a random reference plan
NEAR = 0.4 # real margin below this (times the scale) counts as near the edge, where the barrier is often active
# Hyperparameters changed from scp.yaml's: a small and a large trust region, and a w_dyn low enough that the
# dynamics slack gets used
SETTINGS = {"scp.yaml": {}, "w_prox 0.01": {"w_prox": 0.01}, "w_prox 100": {"w_prox": 100.0}, "w_dyn 1": {"w_dyn": 1.0}}

def subproblem_cost(ctrl, matrices, z, u, z_ref, u_ref, hyperparams):
    # The subproblem's objective at a solution: the linearised cost (its slacks equal the L1 penalties there) plus the
    # trust region, with this setting's weights
    saved, ctrl.hyperparams = ctrl.hyperparams, hyperparams
    cost = ctrl.linear_cost_fn(matrices, z, u)
    ctrl.hyperparams = saved
    return cost + hyperparams["w_prox"] * (np.sum((z[1:] - z_ref[1:]) ** 2) + np.sum((u - u_ref) ** 2))

def check(wm_cfg, common_cfg, scp_cfg):
    with h5py.File(ROOT / yaml.safe_load(open(ROOT / wm_cfg))["dataset_filename"], "r") as f:
        obs = f["observations"][:]
    cfg = build_cfg(wm_cfg, common_cfg, scp_cfg, obs.shape[-1])
    agent = make_agent(cfg, obs.shape[-1])
    ctrl = agent.controller
    r_fn, v_fn, f_fn, Q_fn = agent.wm_networks
    args = (ctrl.horizon, ctrl.d_z, ctrl.d_u, cfg["discount"], cfg["rho_u"], *cfg["action_bounds"])
    solvers = {"cvxpy": SCPSolver(*args), "direct": DirectSCPSolver(*args)}
    rng = np.random.default_rng(0)

    margin = margin_labels(obs, cfg["margin_constraints"])[:, 0] / cfg["margin_constraints"][0]["scale"]
    near = rng.choice(np.flatnonzero(margin < NEAR), N_STATES // 2, replace=False)
    rows = np.concatenate([near, rng.choice(len(obs), N_STATES - len(near), replace=False)])

    results = {name: {"du": [], "dz": [], "dcost": [], "fails": 0, "active": 0, "slack": 0,
                      "time": {k: [] for k in solvers}} for name in SETTINGS}
    with enable_x64():
        for row in rows:
            # A random reference plan from a dataset state, rolled out and linearised as SCP does
            z_c = jnp.asarray(agent.world_model.encoder(jnp.asarray(obs[row])), dtype=jnp.float64)
            u_ref = rng.uniform(*cfg["action_bounds"], (ctrl.horizon, ctrl.d_u))
            z_ref, _, _ = ctrl.rollout(z_c, u_ref, agent.wm_networks)
            matrices = tuple(np.asarray(m) for m in ctrl.jac_extractor_fn(
                z_ref=jnp.array(z_ref), u_ref=jnp.array(u_ref), r_fn=r_fn, v_fn=v_fn, f_fn=f_fn, Q_fn=Q_fn))
            for name, setting in SETTINGS.items():
                hyperparams = {**ctrl.hyperparams, **setting}
                out = {}
                for k, solver in solvers.items():
                    start = time.perf_counter()
                    out[k] = solver.solve_problem(matrices, np.asarray(z_c), z_ref, u_ref, hyperparams)
                    results[name]["time"][k].append(time.perf_counter() - start)
                (u_a, z_a, ok_a), (u_b, z_b, ok_b) = out["cvxpy"], out["direct"]
                if not (ok_a and ok_b):
                    results[name]["fails"] += 1
                    continue
                cost_a, cost_b = (subproblem_cost(ctrl, matrices, z, u, z_ref, u_ref, hyperparams)
                                  for z, u in ((z_a, u_a), (z_b, u_b)))
                results[name]["du"].append(np.max(np.abs(u_a - u_b)))
                results[name]["dz"].append(np.max(np.abs(z_a - z_b)))
                results[name]["dcost"].append(abs(cost_a - cost_b) / max(abs(cost_a), 1e-8))
                dyn_slack, _, _, safe_multiplier = solvers["direct"].slack_and_multipliers()
                results[name]["active"] += safe_multiplier > 1e-6 # the barrier constraint binds
                results[name]["slack"] += dyn_slack > 1e-6 # the dynamics slack is used

    print(f"\n{cfg['data_dir']}: cvxpy against the direct solver on {N_STATES} subproblems per setting")
    for name, res in results.items():
        t = {k: 1e3 * np.median(v[1:]) for k, v in res["time"].items()} # the first solve includes setup
        print(f"  {name:18s} largest difference: actions {max(res['du']):.1e}, latents {max(res['dz']):.1e}, "
              f"objective {max(res['dcost']):.1e} (relative) | barrier binding in {res['active']}, dynamics slack "
              f"used in {res['slack']} | solver failures {res['fails']} | median solve cvxpy {t['cvxpy']:.1f} ms, "
              f"direct {t['direct']:.1f} ms")

if __name__ == "__main__":
    check("configs/wmTraining/pg1_c_1_5_model.yaml", "configs/Planner/common.yaml", "configs/Planner/scp.yaml")
