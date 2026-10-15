import sys
import csv
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import yaml
import orbax.checkpoint as ocp
import safety_gymnasium as sgym

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from ..safe_jepa.agent.Agent import SafeJEPAAgent
from .train_world_model import resolve_dims

def load_cfg(*paths):
    # Merge YAMLs in order; refuse a key defined in more than one file so nothing is silently overridden
    cfg = {}
    for path in paths:
        with open(path) as f:
            new = yaml.safe_load(f)
        clash = cfg.keys() & new.keys()
        if clash:
            raise ValueError(f"{path} redefines {sorted(clash)}")
        cfg.update(new)
    return cfg

def make_agent(wm_cfg: str, common_cfg: str, planner_cfg: str, obs_dim: int):
    # World model config + shared planner settings + the chosen controller
    cfg = load_cfg(ROOT / wm_cfg, ROOT / common_cfg, ROOT / planner_cfg)
    cfg = resolve_dims(cfg, obs_dim, cfg["d_u"])

    # Placeholder obs stats are fine: restoring the checkpoint overwrites them with the saved ones
    agent = SafeJEPAAgent(cfg, jnp.zeros(obs_dim), jnp.ones(obs_dim), rngs=nnx.Rngs(cfg["seed"]))

    # Load the trained world model saved by train_world_model.py
    ckpt_path = ROOT / "src/save_data" / cfg["data_dir"] / "Model/state"
    _, target = nnx.split(agent.world_model)
    nnx.update(agent.world_model, ocp.StandardCheckpointer().restore(ckpt_path, target))
    return agent, cfg

def run_sim(agent: SafeJEPAAgent, env: str, seed: int, max_steps: int) -> dict:
    # Initialise episode and agent
    obs, _ = env.reset(seed=seed)
    agent.reset()
    agent.key = jax.random.key(seed)

    # Initialise termination flag
    done = False

    # Initialise logs
    ep_return = 0.0
    plan_times = []
    actions = []
    rewards = []
    viols = []
    safety_costs = []
    obs_log = []
    next_obs_log = []
    done_log = []
    
    for t in range(max_steps):
        # Time action selection for analysis
        start = time.perf_counter()
        u = agent.act(obs)
        plan_times.append(time.perf_counter() - start)

        # Step forward with agents action
        next_obs, reward, cost, terminated, truncated, _ = env.step(u)
        viol = (cost > 0)
        done = (terminated or truncated)

        # Log results
        rewards.append(reward)
        viols.append(viol)
        safety_costs.append(cost)
        actions.append(u)
        obs_log.append(obs)
        next_obs_log.append(next_obs)
        ep_return += reward

        # If crashed or terminated end the episode
        if done or viol:
            break
        else:
            obs = next_obs

    summary = {
        "seed": seed,
        "steps": t + 1,
        "return": float(ep_return),
        "violated": bool(viols[-1]),
        "plan_time_s": float(np.median(plan_times))
    }

    trajectory =  {
        "actions": np.array(actions),
        "obs": np.array(obs_log),
        "next_obs": np.array(next_obs_log),
        "violations": np.array(viols),
        "safety_costs": np.array(safety_costs),
        "rewards": np.array(rewards),
        "plan_times": np.array(plan_times),
        "done_log": np.array(done_log)
    }
    return summary, trajectory

def run_experiment(
        seeds, 
        wm_cfg_filename, 
        controller_cfg_filename, 
        common_cfg_filename, 
        env_name: str, 
        render_mode: str = None
    ):
    # Setup env and agent
    env = sgym.make(env_name, render_mode=render_mode)

    # Setup agent
    agent, cfg = make_agent(wm_cfg_filename, common_cfg_filename, controller_cfg_filename, env.observation_space.shape[0])

    # Setup directories
    out_dir = ROOT / "src/save_data/experiments" / cfg["data_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cfg['controller']}_seeds_{seeds[0]}-{seeds[-1]}.csv"

    # Save the exact merged config next to the results
    with open(out_dir / f"{cfg['controller']}_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    rows = []
    for seed in seeds:
        summary, trajectory = run_sim(agent, env, seed, cfg["sim_max_steps"])
        rows.append({"world_model": cfg["data_dir"], "controller": cfg["controller"], **summary})
        print(rows[-1], flush=True)

        np.savez_compressed(out_dir / f"{cfg['controller']}_seed_{seed}.npz", **trajectory)

        # Rewrite after every episode so a killed job keeps the episodes it finished
        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

if __name__ == "__main__":
    run_experiment(
        seeds=list(range(20)),
        wm_cfg_filename="configs/wmTraining/standard_model.yaml",
        controller_cfg_filename="configs/Planner/cem.yaml",
        common_cfg_filename="configs/Planner/common.yaml",
        env_name="SafetyPointGoal1-v0"
    )