import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import h5py
import yaml
from functools import partial
from tqdm import tqdm

from safe_wm.World_Model.WorldModel import WorldModel

@partial(jax.jit, static_argnames=("batch_size",))
def _sample(data_dict, key, batch_size):
    n = data_dict["observations"].shape[0]
    indices = jax.random.randint(key, (batch_size,), 0, n)
    return {k: v[indices] for k, v in data_dict.items()}

def get_batches(data_dict, batch_size, N, key):
    for i in range(N):
        yield _sample(data_dict, jax.random.fold_in(key, i), batch_size)

def main(cfg_filename):
    # Load training config
    with open(cfg_filename, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Initialise datadict
    data_dict = {}

    # Load dataset (assume dataset has been processed)
    with h5py.File(cfg['dataset_filename'], 'r') as f:
        for key in list(f.keys()):
            data_dict[key] = f[key][:]

    # Setup world model
    world_model = WorldModel(
        1, 1, 1, 1, 0.1, rngs=1
    )

    # Run training loop
    for epoch in tqdm(range(cfg["epochs"]), desc=f"Epochs", leave=True):
        for batch in tqdm(get_batches(data_dict, cfg["batch_size"], cfg["num_batches"], key), desc=f"Batches", leave=False):
            obs = batch["observations"]
            next_obs = batch["next_observations"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            safety_costs = batch["costs"]
            terminals = batch["terminals"]
            world_model.train_step(
                obs, next_obs, actions, rewards, safety_costs, terminals
            )

    return 0

if __name__ == "__main__":
    main()