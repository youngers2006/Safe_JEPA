import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from analyse_episodes import load_world_model
from train_world_model import episode_ids

N_WINDOWS = 2000 # windows of H + 1 consecutive dataset rows from one trajectory
H = 10 # rollout steps checked (the planning horizon)
N_ACTIONS = 64 # random actions tried from each latent

@nnx.jit
def _encode(world_model, obs):
    return world_model.target_encoder(obs)

@nnx.jit
def _step(world_model, z, u):
    return world_model.dynamics(z, u)

def check(cfg_path):
    cfg = yaml.safe_load(open(ROOT / cfg_path))
    world_model, data = load_world_model(cfg)
    ep = episode_ids(data)
    rng = np.random.default_rng(0)
    starts = rng.choice(np.flatnonzero(ep[:-H] == ep[H:]), N_WINDOWS, replace=False)
    z_real = [np.asarray(_encode(world_model, jnp.asarray(data["observations"][starts + k]))) for k in range(H + 1)]

    # Rollout error against predicting no change (z_k = z_0), and with the windows' action sequences shuffled between
    # windows: a model that uses the action does clearly worse with the wrong actions
    def rollout_error(rows):
        z, err = jnp.asarray(z_real[0]), []
        for k in range(H):
            z = _step(world_model, z, jnp.asarray(data["actions"][rows + k]))
            err.append(np.mean((np.asarray(z) - z_real[k + 1]) ** 2))
        return err
    err_model, err_shuffled = rollout_error(starts), rollout_error(rng.permutation(starts))
    err_still = [np.mean((z_real[0] - z_real[k + 1]) ** 2) for k in range(H)]
    print(f"\n{cfg['data_dir']}: {N_WINDOWS} windows of {H} steps")
    print("  rollout MSE per latent dim, model vs predicting no change vs model with shuffled actions: " + ", ".join(
        f"step {k + 1} {err_model[k]:.4f} vs {err_still[k]:.4f} vs {err_shuffled[k]:.4f}" for k in (0, 1, 4, H - 1)))

    # How much the predicted next latent depends on the action: its spread over random actions from the same latents
    z0 = jnp.asarray(z_real[0][:500])