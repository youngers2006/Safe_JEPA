import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import yaml
import orbax.checkpoint as ocp
import safety_gymnasium as sgym
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from safe_jepa.agent.Agent import SafeJEPAAgent
from train_world_model import resolve_dims
from controller_experiment import load_cfg, run_sim   # episode loop reused as-is

class RecordingAgent:
    """Proxy that records the planner's info dict on every act().

    run_sim only touches .act, .reset and .key, so wrapping the agent here lets
    the existing episode loop be reused instead of forked for the new metrics.
    """

    def __init__(self, agent):
        object.__setattr__(self, "_agent", agent)
        object.__setattr__(self, "info_log", [])

    def act(self, obs):
        u = self._agent.act(obs)
        self.info_log.append(dict(getattr(self._agent, "last_info", None) or {}))
        return u

    def reset(self):
        self.info_log.clear()
        return self._agent.reset()

    def __getattr__(self, name):          # only called when normal lookup fails
        return getattr(self._agent, name)

    def __setattr__(self, name, value):   # run_sim sets agent.key
        setattr(self._agent, name, value)

def build_cfg(wm_cfg, common_cfg, planner_cfg, obs_dim, overrides=None):
    """load_cfg refuses a key defined twice, so sweeps override after the merge."""
    cfg = load_cfg(ROOT / wm_cfg, ROOT / common_cfg, ROOT / planner_cfg)
    for k, v in (overrides or {}).items():
        if k not in cfg:
            raise KeyError(f"override {k!r} is not an existing config key")
        cfg[k] = v
    return resolve_dims(cfg, obs_dim, cfg["d_u"])

def make_agent(cfg, obs_dim):
    """Same body as controller_experiment.make_agent, minus the cfg building.
    If that one is ever changed, change this too."""
    agent = SafeJEPAAgent(cfg, jnp.zeros(obs_dim), jnp.ones(obs_dim),
                          rngs=nnx.Rngs(cfg["seed"]))
    ckpt_path = ROOT / "src/save_data" / cfg["data_dir"] / "Model/state"
    _, target = nnx.split(agent.world_model)
    nnx.update(agent.world_model, ocp.StandardCheckpointer().restore(ckpt_path, target))
    return agent

def warmup(agent, env, seed):
    """One discarded act() so JIT compilation stays out of the timed loop.
    Without it the first call lands in plan_time_p99."""
    obs, _ = env.reset(seed=seed)
    agent.reset()
    agent.key = jax.random.key(seed)
    agent.act(obs)
    agent.reset()

def _mean_of(info_log, key):
    vals = [float(d[key]) for d in info_log if d.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")

def summarise(summary, trajectory, info_log, max_steps):
    out = dict(summary)
    steps = summary["steps"]

    # Right-censored for Kaplan-Meier: survived to the cap without crashing.
    out["censored"] = bool(not summary["violated"] and steps >= max_steps)
    out["reward_rate"] = summary["return"] / steps

    pt = trajectory["plan_times"]
    out["plan_time_med"] = float(np.median(pt))
    out["plan_time_p99"] = float(np.percentile(pt, 99))

    out["feasible_frac_first"] = _mean_of(info_log, "feasible_frac_first")
    out["feasible_frac_last"] = _mean_of(info_log, "feasible_frac_last")
    out["mean_feasible_rate"] = _mean_of(info_log, "mean_feasible")
    out["rejection_gap"] = _mean_of(info_log, "rejection_gap")
    out["scp_iters"] = _mean_of(info_log, "iterations")
    out["scp_last_rejected_rate"] = _mean_of(info_log, "last_rejected")
    sr = [str(d["stop_reason"]) for d in info_log if "stop_reason" in d]
    out["scp_converged_rate"] = float(np.mean([r != "max_iters" for r in sr])) if sr else float("nan")
    out["pred_return"] = _mean_of(info_log, "pred_return")
    out["pred_max_risk"] = _mean_of(info_log, "pred_max_risk")
    out["pred_feasible_rate"] = _mean_of(info_log, "pred_feasible")

    af = [bool(d["any_feasible"]) for d in info_log if "any_feasible" in d]
    out["fallback_rate"] = float(1.0 - np.mean(af)) if af else float("nan")
    return out

def _info_arrays(info_log):
    out = {}
    for k in {k for d in info_log for k in d}:
        arr = np.stack([np.asarray(d[k]) for d in info_log])
        out[f"info_{k}"] = arr.astype(float) if arr.dtype.kind in "biuf" else arr
    return out

def run_point(seeds, wm_cfg, common_cfg, planner_cfg, env_name,
              overrides, tag, render_mode=None):
    env = sgym.make(env_name, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]

    cfg = build_cfg(wm_cfg, common_cfg, planner_cfg, obs_dim, overrides)
    agent = RecordingAgent(make_agent(cfg, obs_dim))

    out_dir = ROOT / "src/save_data/experiments" / cfg["data_dir"] / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{cfg['controller']}_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    out_file = out_dir / f"{cfg['controller']}_seeds_{seeds[0]}-{seeds[-1]}.csv"

    warmup(agent, env, seeds[0])

    rows = []
    for seed in tqdm(seeds, desc=f"{tag} {cfg['controller']}", leave=True):
        summary, trajectory = run_sim(agent, env, seed, cfg["sim_max_steps"])
        rows.append({
            "world_model": cfg["data_dir"],
            "controller": cfg["controller"],
            "margin_eps": cfg["margin_eps"],
            "barrier_alpha": cfg["barrier_alpha"],
            "num_seqs": cfg.get("num_seqs", float("nan")),
            **summarise(summary, trajectory, agent.info_log, cfg["sim_max_steps"]),
        })
        print(rows[-1], flush=True)

        np.savez_compressed(out_dir / f"{cfg['controller']}_seed_{seed}.npz",
                            **trajectory, **_info_arrays(agent.info_log))

        with open(out_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    env.close()
    return rows