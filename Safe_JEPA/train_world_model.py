import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import h5py
import yaml

from safe_wm.World_Model.WorldModel import WorldModel

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

    # Batch Dataset 

    
    return 0



if __name__ == "__main__":
    main()