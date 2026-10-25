"""Measure the constraint multipliers of SCP's subproblems along a closed-loop episode, to choose w_slack.

The L1 slack penalty is exact (its slack stays at zero whenever a feasible plan exists) only when w_slack is above every
multiplier, and the penalty caps each multiplier at w_slack: a multiplier sitting at w_slack means that slack is in use.
For each w_slack given, this runs one episode and reports, from the final subproblem of every step, how often each slack
is in use and the largest multipliers on the steps where it is not. Choose w_slack above those. A step whose safety slack
is in use at every w_slack has no feasible plan within SCP's reach at that threshold.

usage: python src/scripts/measure_multipliers.py configs/wmTraining/pg1_c_1_5_model.yaml --tau 0.02 --w_slack 1 10 100
"""
import argparse

import jax
import numpy as np

from sweep_runner import build_cfg, make_agent, sgym

SLACK_TOL = 1e-7

def last_solve_recorder(solver, log):
    # Wrap solve_problem so the slack and multipliers of the latest successful subproblem are kept in log
    dyn = [c for c in solver.problem.constraints if any(v is solver.mu_p for v in c.variables())]
    safe = [c for c in solver.problem.constraints if any(v is solver.nu for v in c.variables())]
    solve = solver.solve_problem

    def recorded(*args, **kwargs):
        u, z, ok = solve(*args, **kwargs)
        if ok:
            log["last"] = (float(np.max(solver.mu_p.value + solver.mu_n.value)), float(np.max(solver.nu.value)),
                           max(float(np.max(np.abs(c.dual_value))) for c in dyn),
                           max(float(np.max(c.dual_value)) for c in safe))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("wm_cfg")
    parser.add_argument("--common_cfg", default="configs/Planner/common.yaml")
    parser.add_argument("--scp_cfg", default="configs/Planner/scp.yaml")
    parser.add_argument("--env", default="SafetyPointGoal1-v0")
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--w_slack", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = sgym.make(args.env)
    for w in args.w_slack:
        cfg = build_cfg(args.wm_cfg, args.common_cfg, args.scp_cfg, env.observation_space.shape[0],
                        {"safety_threshold": args.tau, "w_slack": w})
        rows = run_episode(cfg, env, args.seed, args.steps)
        slack_dyn, slack_safe, mult_dyn, mult_safe, max_risk, feasible = rows.T
        free_dyn, free_safe = slack_dyn <= SLACK_TOL, slack_safe <= SLACK_TOL
        largest = lambda m, free: f"{m[free].max():.3g}" if free.any() else "-"
        print(f"w_slack {w:g}, tau {args.tau:g}: {len(rows)} steps | dynamics slack in use on {np.mean(~free_dyn):.2f} of steps, "
              f"largest dynamics multiplier without it {largest(mult_dyn, free_dyn)} | safety slack in use on "
              f"{np.mean(~free_safe):.2f} of steps, largest safety multiplier without it {largest(mult_safe, free_safe)} | "
              f"planned max risk median {np.median(max_risk):.3f}, feasible plans {np.mean(feasible):.2f}", flush=True)
    env.close()

if __name__ == "__main__":
    main()
