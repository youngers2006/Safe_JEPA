import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import h5py
import yaml

from safe_wm.World_Model.WorldModel import WorldModel

def get_batches(dataset_size, batch_size, key: np.random.Generator):
    indices = np.arange(dataset_size)
    key.shuffle(indices)
    for i in range(0, dataset_size - batch_size + 1, batch_size):
        yield indices[i:i + batch_size]

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
    world_model = WorldModel(1, 1, 1, 1, 0.1, rngs=1)

    # Run training loop
    for batch in get_batches(...):
        world_model.train_step(obs, next_obs, action, reward, safety_cost, done)
    return 0



if __name__ == "__main__":
    main()