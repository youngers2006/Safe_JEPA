import sys
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import orbax.checkpoint as ocp
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from safe_jepa.world_model.world_model import WorldModel
from train_world_model import resolve_dims

def load_trained(cfg_filename):
    # Rebuild the world model from its training config and restore the saved weights (as sweep_runner does)
    cfg = yaml.load(open(ROOT / cfg_filename), Loader=yaml.FullLoader)
    with h5py.File(ROOT / cfg["dataset_filename"], 'r') as f:
        data = {k: f[k][:] for k in f.keys()}
    cfg = resolve_dims(cfg, data["observations"].shape[-1], data["actions"].shape[-1])
    obs_dim = data["observations"].shape[-1]
    world_model = WorldModel(cfg, jnp.zeros(obs_dim), jnp.ones(obs_dim), rngs=nnx.Rngs(cfg["seed"]))
    _, target = nnx.split(world_model)
    ckpt_path = ROOT / "src/save_data" / cfg["data_dir"] / "Model/state"
    nnx.update(world_model, ocp.StandardCheckpointer().restore(ckpt_path, target))
    return world_model, cfg, data

def local_slopes(fn, x):
    # Spectral norm of the Jacobian at each input = how steep the network is there (its Lipschitz constant is the max)
    jac = jax.vmap(jax.jacfwd(lambda xi: jnp.atleast_1d(fn(xi[None])[0])))(x)
    return np.asarray(jnp.linalg.svd(jac, compute_uv=False)[:, 0])

def main(cfg_filename, n_points=4096):
    world_model, cfg, data = load_trained(cfg_filename)
    idx = np.random.default_rng(0).choice(len(data["observations"]), n_points, replace=False)
    obs = jnp.asarray(data["observations"][idx])
    act = jnp.asarray(data["actions"][idx])
    d_z = cfg["d_latent"]

    # Latent states and actions at real data points
    z = world_model.encoder(obs)
    zu = jnp.concatenate([z, act], axis=-1)

    nets = {
        "dynamics": (lambda x: world_model.dynamics(x[:, :d_z], x[:, d_z:]), zu, "DynamicsParams"),
        "reward": (lambda x: world_model.reward_fn(x[:, :d_z], x[:, d_z:]), zu, "RewardParams"),
        "value": (lambda x: world_model.value_fn(x), z, "ValueParams"),
        "safety_critic": (lambda x: world_model.safety_critic(x[:, :d_z], x[:, d_z:]), zu, "SafetyCriticParams"),
    }
    print(f"{cfg_filename} ({cfg['data_dir']}), local slope over {n_points} data points:")
    for name, (fn, x, key) in nets.items():
        s = local_slopes(fn, x)
        print(f"  {name:14s} bound {cfg[key]['lipschitz_bound']:<5} | median {np.median(s):.3f}  "
              f"95% {np.percentile(s, 95):.3f}  max {s.max():.3f}")

    # Dynamics fit in this model's latent space, against just predicting next_z = z
    next_z_target = world_model.target_encoder(jnp.asarray(data["next_observations"][idx]))
    dyn_mse = float(jnp.mean((world_model.dynamics(z, act) - next_z_target) ** 2))
    copy_mse = float(jnp.mean((z - next_z_target) ** 2))
    print(f"  dynamics MSE {dyn_mse:.5f} vs copying z {copy_mse:.5f} (ratio {dyn_mse / copy_mse:.3f}, < 1 beats copying)")

if __name__ == "__main__":
    for cfg_filename in sys.argv[1:]:
        main(cfg_filename)
