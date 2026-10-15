import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import flax.nnx as nnx
from functools import partial

from .evaluate_sequences import evaluate_sequences, _evaluate_sequences

class SamplingController():
    def __init__(self, cfg: dict):
        # Sampling method
        if cfg["controller"] not in ("CEM", "MPPI"):
            raise ValueError(f"No sampling methods of type: {cfg['sample_method']}")
        self.sample_method = cfg["controller"]

        # Problem dimensions
        self.horizon = cfg["horizon"]
        self.d_u = cfg["d_u"]
        self.u_bounds = cfg["action_bounds"]

        # Objective
        self.safety_threshold = cfg["safety_threshold"]
        self.discount = cfg["discount"]
        self.rho_u = cfg["rho_u"]

        # Optimisation loop settings
        self.top_k = cfg["top_k"]
        self.temperature = cfg["temperature"]
        self.optimisation_iters = cfg["opt_iters"]
        self.num_seqs = cfg["num_seqs"]

        # Sampling distribution variance
        self.var_init = (cfg["init_std_frac"] * (self.u_bounds[1] - self.u_bounds[0])) ** 2
        self.var_min = cfg["var_min"]
        self.var_max = cfg["var_max"]

    def sample_sequences(self, q: dict, num_seqs: int, key):
        eps = jax.random.normal(key, (num_seqs, *q["mean"].shape))
        seqs = q["mean"] + eps * jnp.sqrt(q["variance"])
        return jnp.clip(seqs, self.u_bounds[0], self.u_bounds[1])

    def rank_policies(self, r_seq: jax.Array, feasible: jax.Array, viols: jax.Array, top_k: int):
        # If any trajectories are feasible then penalise infeasibility heavily, if all trajectories are infeasible then score by total violations
        any_feas = feasible.any()
        scores = jnp.where(
            any_feas,
            jnp.where(feasible, r_seq, -1e12),
            -jnp.sum(viols, axis=-1)
        )

        # Select top-k elites
        elite_vals, elite_idx = lax.top_k(scores, top_k)

        # Create weightings for update based on score
        w = self.weight_fn(elite_vals, self.temperature)
        mask = feasible[elite_idx].astype(w.dtype)
        w = jnp.where(any_feas, w * mask, w)
        w = w / jnp.maximum(w.sum(), 1e-8)

        return elite_idx, w, any_feas

    def weight_fn(self, values, temperature):
        # MPPI: exponential weighting of elites by score
        if self.sample_method == "MPPI":
            return jnp.exp((values - jnp.max(values)) / temperature)
        # CEM: uniform weighting
        return jnp.ones(shape=values.shape, dtype=values.dtype)

    def init_params(self):
        params = {
            "mean": jnp.zeros((self.horizon, self.d_u)),
            "variance": jnp.full((self.horizon, self.d_u), self.var_init)
        }
        return params

    def shift_q(self, q):
        return {
            "mean": jnp.concatenate([q["mean"][1:], q["mean"][-1:]], axis=0),
            "variance": jnp.full_like(q["variance"], self.var_init)
        }

    def update_params(self, q, u_seqs, elite_idx, w):
        elites = u_seqs[elite_idx] # (k, N, d_u)
        wk = w[:, None, None] # (k, 1, 1)
        mean = jnp.sum(wk * elites, axis=0) # (N, d_u)
        var = jnp.sum(wk * jnp.square(elites - mean), axis=0) # (N, d_u)
        return {"mean": mean, "variance": jnp.clip(var, self.var_min, self.var_max)}

    def step(self, z_c, wm_networks, key, q_prev = None) -> jax.Array:
        q0 = self.init_params() if q_prev is None else self.shift_q(q_prev)
        return self.plan(z_c, wm_networks, key, q0)

    @partial(nnx.jit, static_argnums=0)
    def plan(self, z_c, wm_networks, key, q0):
        def scan_step(carry, _):
            q, key = carry
            key, sub = jax.random.split(key)

            action_seqs = self.sample_sequences(q, self.num_seqs, sub)
            seq_reward, seq_feasible, seq_viols = evaluate_sequences(
                z_c, action_seqs, *wm_networks, self.safety_threshold, self.discount, self.rho_u
            )
            elite_idx, weights, any_feas = self.rank_policies(
                seq_reward, seq_feasible, seq_viols, self.top_k
            )
            q = self.update_params(
                q, action_seqs, elite_idx, weights
            )
            return (q, key), (any_feas, action_seqs[elite_idx[0]])

        (q_best, _), (feasibility_history, best_hist) = lax.scan(
            scan_step, (q0, key), None, length=self.optimisation_iters 
        )

        _, mean_feas, _ = _evaluate_sequences(
            z_c, q_best["mean"][None, :, :], *wm_networks, self.safety_threshold, self.discount, self.rho_u
        )
        mean_ok = mean_feas[0]
        u_seq = jnp.where(mean_ok, q_best["mean"], best_hist[-1])

        return u_seq[0], q_best, {"any_feasible": feasibility_history[-1], "mean_feasible": mean_ok}
        