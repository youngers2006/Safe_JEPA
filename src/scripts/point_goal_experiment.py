import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import yaml

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