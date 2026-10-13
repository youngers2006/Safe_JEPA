import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import flax.nnx as nnx

@nnx.jit
def evaluate_sequences(z0, u_seq, r_fn, v_fn, f_fn, Q_fn, safety_threshold):
    # Get computation graphs of the networks for lax scan
    r_graph, r_state = nnx.split(r_fn)
    f_graph, f_state = nnx.split(f_fn)
    Q_graph, Q_state = nnx.split(Q_fn)

    # setup lax scan
    def scan_step(z_current, u_current):
        z_next = nnx.merge(f_graph, f_state)(
            z_current, u_current, update_spectral_norm=False
        )
        reward = nnx.merge(r_graph, r_state)(
            z_current, u_current, update_spectral_norm=False
        )
        q_safe = nnx.merge(Q_graph, Q_state)(
            z_current, u_current, update_spectral_norm=False
        )
        return z_next, (z_next, reward, q_safe)

    # Scan the function and rollout trajectory in the world model
    z_terminal, (z_traj, rewards, q_safe) = lax.scan(scan_step, z0, u_seq) # (leaves us with J sequences)

    # Total reward for each trajectory
    traj_reward = jnp.sum(rewards, axis=0)

    # Total Feasibility
    violations = (q_safe > safety_threshold)
    traj_violations = jnp.sum(violations, axis=-1)
    traj_feasibility = (traj_violations == 0.0)

    return traj_reward, traj_feasibility, violations