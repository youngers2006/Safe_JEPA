"""Measure the constraint multipliers of SCP's subproblems along a closed-loop episode, to choose w_dyn and w_safe.

An L1 slack penalty is exact (its slack stays at zero whenever a feasible plan exists) only when its weight is above
every multiplier of its constraints, and the penalty caps each multiplier at the weight: a multiplier sitting at the
weight means that slack is in use. For each pair of weights given, this runs one episode and reports, from the final
subproblem of every step, how often each slack is in use and the largest multipliers on the steps where it is not.
Choose w_dyn above the dynamics multipliers and w_safe above the safety ones. A step whose safety slack is in use at
every w_safe has no feasible plan within SCP's reach at that threshold.

Set the settings below, then: python src/scripts/measure_multipliers.py
"""
import itertools

import jax
import numpy as np

from sweep_runner import build_cfg, make_agent, sgym

SLACK_TOL = 1e-7
WM_CFG = "configs/wmTraining/pg1_c_1_5_model.yaml"
COMMON = "configs/Planner/common.yaml"
SCP_CFG = "configs/Planner/scp.yaml"
ENV = "SafetyPointGoal1-v0"
TAU = 0.02
W_DYN = [1.0]
W_SAFE = [1.0, 10.0, 100.0]
STEPS = 200
SEED = 0

def last_solve_recorder(solver, log):
    # Wrap solve_problem so the slack and multipliers of the latest successful subproblem are kept in log
    if hasattr(solver, "slack_and_multipliers"):
        read = solver.slack_and_multipliers
    else:
        # cvxpy formulation: the dynamics constraints are those with mu_p, the safety constraints those with nu
        dyn = [c for c in solver.problem.constraints if any(v is solver.mu_p for v in c.variables())]
        safe = [c for c in solver.problem.constraints if any(v is solver.nu for v in c.variables())]
        read = lambda: (float(np.max(solver.mu_p.value + solver.mu_n.value)), float(np.max(solver.nu.value)),
                        max(float(np.max(np.abs(c.dual_value))) for c in dyn),
                        max(float(np.max(c.dual_value)) for c in safe))
    solve = solver.solve_problem

    def recorded(*args, **kwargs):
        u, z, ok = solve(*args, **kwargs)
        if ok:
            log["last"] = read()
        return u, z, ok
    return recorded

def run_episode(cfg, env, seed, steps):
    agent = make_agent(cfg, env.observation_space.shape[0])
    obs, _ = env.reset(seed=seed)
    agent.reset()
    agent.key = jax.random.key(seed)

    # The agent rebuilt its controller on reset, so wrap the solver of the new one
    log, rows = {}, []
    solver = agent.controller.SCP_solver
    solver.solve_problem = last_solve_recorder(solver, log)
    for _ in range(steps):
        log.pop("last", None)
        u = agent.act(obs)
        if "last" in log:
            rows.append((*log["last"], float(agent.last_info["pred_max_risk"]), bool(agent.last_info["pred_feasible"])))
        obs, _, cost, terminated, truncated, _ = env.step(u)
        if cost > 0 or terminated or truncated:
            break
    return np.array(rows, dtype=float)

def main():
    env = sgym.make(ENV)
    for w_dyn, w_safe in itertools.product(W_DYN, W_SAFE):
        cfg = build_cfg(WM_CFG, COMMON, SCP_CFG, env.observation_space.shape[0],
                        {"safety_threshold": TAU, "w_dyn": w_dyn, "w_safe": w_safe})
        rows = run_episode(cfg, env, SEED, STEPS)
        slack_dyn, slack_safe, mult_dyn, mult_safe, max_risk, feasible = rows.T
        free_dyn, free_safe = slack_dyn <= SLACK_TOL, slack_safe <= SLACK_TOL
        # NaN marks a multiplier the solver could not resolve accurately
        largest = lambda m, free: f"{np.nanmax(m[free]):.3g}" if np.any(free & ~np.isnan(m)) else "-"
        print(f"w_dyn {w_dyn:g}, w_safe {w_safe:g}, tau {TAU:g}: {len(rows)} steps | dynamics slack in use on {np.mean(~free_dyn):.2f} of steps, "
              f"largest dynamics multiplier without it {largest(mult_dyn, free_dyn)} | safety slack in use on "
              f"{np.mean(~free_safe):.2f} of steps, largest safety multiplier without it {largest(mult_safe, free_safe)} | "
              f"planned max risk median {np.median(max_risk):.3f}, feasible plans {np.mean(feasible):.2f}", flush=True)
    env.close()

if __name__ == "__main__":
    main()
