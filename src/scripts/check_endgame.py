import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from sweep_runner import make_agent
from train_world_model import resolve_dims
from analyse_episodes import load_episodes, RESULTS_PATH, TAU, FEAS_TOL, LATE_WINDOW
from safe_jepa.SCP_control.SCPController import SCPController
from safe_jepa.Sampling_control.SamplingController import SamplingController

# In the last LATE_WINDOW steps of each violated SCP episode, re-plan from the visited state with SCP as it was run (its
# warm start rebuilt from the saved plans), with a larger safety weight, and started from CEM's plan, and with CEM and
# MPPI. If the larger weight finds feasible plans the safety penalty was too weak; if starting from CEM, or CEM or MPPI,
# finds them SCP was stuck in a local minimum; if nothing finds them the state was already cornered
W_SAFE_BIG = 1e4
SAMPLER_KEYS = 3 # sampler runs per state, the best kept: a fair search for a feasible plan

def load_cfg(path, obs_dim):
    cfg = yaml.safe_load(open(path))
    return resolve_dims(cfg, obs_dim, cfg["d_u"])

def outcome(risk, ret):
    risk = np.asarray(risk)
    return {"feasible": risk.max() <= TAU + FEAS_TOL, "excess": np.maximum(risk - TAU, 0.0).sum(), "return": float(ret)}

def scp_plan(ctrl, z_c, wm, u_start, w_prox, shift):
    # One SCP step from z_c: shift=True starts from u_start shifted one step, as SCP warm starts from its last plan;
    # shift=False starts from u_start as it is
    ctrl.is_first_step, ctrl.u_prev, ctrl.w_prox_warm = not shift, np.array(u_start), w_prox
    _, info = ctrl.step(z_c, wm)
    return outcome(info["pred_risk"], info["pred_return"]), str(info["stop_reason"])

def sampler_plan(ctrl, z_c, wm, seed):
    # The best of SAMPLER_KEYS runs from a fresh sampling distribution: least excess, then highest return
    best, plan = None, None
    for k in range(SAMPLER_KEYS):
        _, _, info = ctrl.step(z_c, wm, jax.random.key(seed * SAMPLER_KEYS + k))
        out = outcome(info["pred_risk"], info["pred_return"])
        if best is None or (out["excess"], -out["return"]) < (best["excess"], -best["return"]):
            best, plan = out, np.asarray(info["plan"], dtype=np.float64)
    return best, plan

def check(experiment, tags):
    exp_dir = RESULTS_PATH / experiment
    print(f"\n{experiment}: SCP's last {LATE_WINDOW} steps before each violation, re-planned. Share of plans meeting "
          f"the barrier (max risk <= {TAU} + {FEAS_TOL}) and mean total excess over the threshold")
    for tag in tags:
        episodes = [(s, ep) for s, ep in load_episodes(exp_dir / tag, "SCP") if bool(ep["violations"][-1])]
        if not episodes:
            continue
        obs_dim = episodes[0][1]["obs"].shape[-1]
        cfg = load_cfg(exp_dir / tag / "SCP_config.yaml", obs_dim)
        agent = make_agent(cfg, obs_dim)
        wm = agent.wm_networks
        control_cfg = {**cfg, "d_z": cfg["d_latent"], "safety_threshold": 0.0}
        scp = {"SCP as run": SCPController(control_cfg),
               f"SCP, w_safe {W_SAFE_BIG:g}": SCPController({**control_cfg, "w_safe": W_SAFE_BIG}),
               "SCP from CEM's plan": SCPController(control_cfg)}

        # The samplers as configured in this sweep point, or in any other (only their own settings are taken)
        samplers = {}
        for name in ("CEM", "MPPI"):
            paths = [exp_dir / tag / f"{name}_config.yaml"] + sorted(exp_dir.glob(f"*/{name}_config.yaml"))
            path = next((p for p in paths if p.exists()), None)
            if path is not None:
                s_cfg = yaml.safe_load(open(path))
                assert s_cfg["horizon"] == cfg["horizon"], f"{name} plans over a different horizon"
                samplers[name] = SamplingController({**s_cfg, "safety_threshold": 0.0})

        results = {name: [] for name in [*scp, *samplers]}
        run_over, replay_match = [], []
        for seed, ep in tqdm(episodes, desc=tag, leave=False):
            T = len(ep["actions"])
            for t in range(max(T - LATE_WINDOW, 0), T):
                z_c = agent.world_model.encoder(jnp.asarray(ep["obs"][t]))
                # SCP's warm start at step t: its plan from step t - 1, and the trust region it ended that step with
                u_prev = ep["info_plan"][t - 1] if t > 0 else np.zeros_like(ep["info_plan"][0])
                w_prox = (float(ep["info_w_prox_final"][t - 1]) if t > 0 and "info_w_prox_final" in ep
                          else cfg["w_prox"])
                run_over.append(ep["info_pred_risk"][t].max() > TAU + FEAS_TOL)

                cem, cem_plan = (sampler_plan(samplers["CEM"], z_c, wm, seed * T + t) if "CEM" in samplers
                                 else (None, None))
                for name, ctrl in scp.items():
                    if name == "SCP from CEM's plan":
                        if cem_plan is None:
                            continue
                        out, _ = scp_plan(ctrl, z_c, wm, cem_plan, w_prox, shift=False)
                    else:
                        out, stop = scp_plan(ctrl, z_c, wm, u_prev, w_prox, shift=t > 0)
                        if name == "SCP as run":
                            replay_match.append(stop == str(ep["info_stop_reason"][t]))
                    results[name].append(out)
                if cem is not None:
                    results["CEM"].append(cem)
                if "MPPI" in samplers:
                    results["MPPI"].append(sampler_plan(samplers["MPPI"], z_c, wm, seed * T + t)[0])

        run_over = np.array(run_over)
        print(f"{tag}: {len(run_over)} steps from {len(episodes)} violated episodes, {run_over.sum()} of them with the "
              f"run's plan over the threshold; the replay reproduces the run's SCP stop reason on "
              f"{np.mean(replay_match):.0%} of steps")
        print(f"  {'':24s} {'all steps':>16s}   {'run over threshold':>18s}")
        for name, outs in results.items():
            if not outs:
                continue
            feas, exc = np.array([o["feasible"] for o in outs]), np.array([o["excess"] for o in outs])
            over = f"{feas[run_over].mean():6.0%} {exc[run_over].mean():9.4f}" if run_over.any() else ""
            print(f"  {name:24s} {feas.mean():6.0%} {exc.mean():9.4f}     {over}")

if __name__ == "__main__":
    EXPERIMENT = "M2_pointgoal1_c_1_5_seed_0"
    TAGS = [f"e1/eps_{eps}" for eps in [0.1, 0.2, 0.3]]
    check(EXPERIMENT, TAGS)