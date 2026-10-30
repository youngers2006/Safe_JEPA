from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import yaml
from pathlib import Path

from ..world_model.world_model import WorldModel
from ..nn.mlp import MarginBarrier
from ..SCP_control.SCPController import SCPController
from ..Sampling_control.SamplingController import SamplingController

ROOT = Path(__file__).resolve().parents[3]

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
            MarginBarrier(
                self.world_model.dynamics, 
                self.world_model.margin_fn, 
                cfg["margin_eps"],
                [c["scale"] for c in cfg["margin_constraints"]],
                cfg["barrier_alpha"]
            ),
        )

        # Store configuration
        self.cfg = cfg

        # PRNG key for randomness
        self.key = rngs()

        # Reset controller and its warm start
        self.reset()

    def make_controller(self):
        control_cfg = {**self.cfg, "d_z": self.cfg["d_latent"], "safety_threshold": 0.0}
        if self.controller_type == "SCP":
            return SCPController(control_cfg)
        return SamplingController(control_cfg)

    def make_seed_sampler(self):
        path = self.cfg.get("seed_sampler")
        if self.controller_type != "SCP" or not path:
            return None
        with open(ROOT / path) as f:
            sampler_cfg = yaml.safe_load(f)
        return SamplingController({**self.cfg, **sampler_cfg, "safety_threshold": 0.0})

    def reset(self) -> None:
        # Refresh controller for a new episode
        self.controller = self.make_controller()
        self.seed_sampler = self.make_seed_sampler()
        self.q_prev = None
        self.last_info = {}

    def act(self, obs: np.ndarray) -> np.ndarray:
        # Encode observation to the current latent state
        z_c = self.world_model.encoder(jnp.asarray(obs))

        # Plan with the chosen controller and return the first action
        if self.controller_type == "SCP":
            u_seed = None
            if self.seed_sampler is not None:
                self.key, sub = jax.random.split(self.key)
                _, self.q_prev, seed_info = self.seed_sampler.step(z_c, self.wm_networks, sub, q_prev=self.q_prev)
                u_seed = np.asarray(seed_info["plan"], dtype=np.float64)
            u, info = self.controller.step(z_c, self.wm_networks, u_seed)
        else:
            self.key, sub = jax.random.split(self.key)
            u, self.q_prev, info = self.controller.step(
                z_c, self.wm_networks, sub, q_prev=self.q_prev
            )
        self.last_info = {k: np.asarray(v) for k, v in info.items()}
        return np.asarray(u)