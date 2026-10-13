from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from .world_model.world_model import WorldModel
# from .SCP_control import SCP_solver

class SafeJEPAAgent:
    def __init__(self, cfg: dict, rngs: nnx.Rngs):
        pass