from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

from ..world_model.world_model import WorldModel
from ..SCP_control.SCPController import SCPController
from ..Sampling_control.SamplingController import SamplingController

class SafeJEPAAgent:
    def __init__(self, cfg: dict, obs_mean: jax.Array, obs_std: jax.Array, *, rngs: nnx.Rngs):
        # Controller choice
        if cfg["controller"] not in ("CEM", "SCP", "MPPI"):
            raise ValueError(f"No controller of type {cfg['controller']}")
        self.controller_type = cfg["controller"]

        # Instantiate world model
        self.world_model = WorldModel(cfg, obs_mean, obs_std, rngs=rngs)

        # Store tuple of references to wm networks
        self.wm_networks = (
            self.world_model.reward_fn,
            self.world_model.value_fn,
            self.world_model.dynamics,
            self.world_model.safety_critic,
        )

        # Store configuration
        self.cfg = cfg

        # PRNG key for randomness
        self.key = rngs()

        # Reset controller and its warm start
        self.reset()

    def make_controller(self):
        control_cfg = {**self.cfg, "d_z": self.cfg["d_latent"]}
        if self.controller_type == "SCP":
            return SCPController(control_cfg)
        return SamplingController(control_cfg)

    def reset(self) -> None:
        # Refresh controller for a new episode
        self.controller = self.make_controller()
        self.q_prev = None
        self.last_info = {}

    def act(self, obs: np.ndarray) -> np.ndarray:
        # Encode observation to the current latent state
        z_c = self.world_model.encoder(jnp.asarray(obs))

        # Plan with the chosen controller and return the first action
        if self.controller_type == "SCP":
            u, info = self.controller.step(z_c, self.wm_networks)
        else:
            self.key, sub = jax.random.split(self.key)
            u, self.q_prev, info = self.controller.step(
                z_c, self.wm_networks, sub, q_prev=self.q_prev
            )
        self.last_info = {k: np.asarray(v) for k, v in info.items()}
        return np.asarray(u)