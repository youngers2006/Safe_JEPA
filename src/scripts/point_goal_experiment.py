import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import yaml
import safety_gymnasium as sgym
import gymnasium as gym

from ..safe_jepa.agent.Agent import SafeJEPAAgent

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

def run_sim(agent: SafeJEPAAgent,env_name: str, render_mode: str, seed: int, max_steps: int):
    # Setup env
    env = sgym(env_name, render_mode)

    # Run sim
    obs, _ = env.reset(seed=seed)
    done = 0

    for t in range(max_steps):
        u = agent.act(obs)
        next_obs, reward, cost, terminated, truncated, _ = env.step(u)
