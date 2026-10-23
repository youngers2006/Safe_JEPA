import jax
import jax.numpy as jnp
from jax import lax
import flax.nnx as nnx

def evaluate_sequences(z0, u_seq, r_fn, v_fn, f_fn, Q_fn, safety_threshold, discount=1.0, rho_u=0.0):
    # Get computation graphs of the networks for lax scan
    r_graph, r_state = nnx.split(r_fn)
    f_graph, f_state = nnx.split(f_fn)
    Q_graph, Q_state = nnx.split(Q_fn)

    # Setup lax scan
    def scan_step(z_current: jax.Array, u_current: jax.Array):
        z_next = nnx.merge(f_graph, f_state)(
            z_current, u_current, update_spectral_norm=False
        )
        reward = nnx.merge(r_graph, r_state)(
            z_current, u_current, update_spectral_norm=False
        )
        q_safe = nnx.merge(Q_graph, Q_state)(z_current, u_current, update_spectral_norm=False)
        return z_next, (z_next, reward, q_safe)

    # extend initial state across all trajectories for scan
    z0 = jnp.broadcast_to(z0, (u_seq.shape[0], z0.shape[-1])) # (dz,) -> (J, dz)

    # Scan the function and rollout trajectory in the world model (swap axes as scan runs across first dim)
    z_terminal, (z_traj, rewards, q_safe) = lax.scan(scan_step, z0, jnp.swapaxes(u_seq, 0, 1)) # returns N, J

    # Predicted return for each trajectory
    horizon = u_seq.shape[1]
    discount_vec = discount ** jnp.arange(horizon)
    pred_return = jnp.sum(discount_vec[:, None] * rewards.squeeze(axis=-1), axis=0)
    pred_return = pred_return + (discount ** horizon) * v_fn(z_terminal, update_spectral_norm=False).squeeze(axis=-1)
    traj_reward = pred_return - rho_u * jnp.sum(u_seq ** 2, axis=(1, 2))

    # Total Feasibility
    violations = (q_safe.squeeze(axis=-1) > safety_threshold).T # (J, N)
    traj_violations = jnp.sum(violations, axis=-1) # (J,)
    traj_feasibility = (traj_violations == 0.0) # (J,)

    predictions = {
        "return": pred_return, # (J,)
        "rewards": rewards.squeeze(axis=-1).T, # (J, N)
        "risk": q_safe.squeeze(axis=-1).T, # (J, N)
        "latents": jnp.swapaxes(z_traj, 0, 1) # (J, N, dz), SCP's reference trajectory
    }

    return traj_reward, traj_feasibility, violations, predictions

def plan_summary(u_seq, feasible, predictions):
    # Model-predicted outcome of the one plan a planner executes (u_seq: (N, d_u), predictions indexed to that plan).
    # Logged so planners and world models can be compared on what the optimiser sees, free of model error
    return {
        "pred_return": predictions["return"],
        "pred_max_risk": jnp.max(predictions["risk"]),
        "pred_feasible": feasible,
        "pred_rewards": predictions["rewards"],
        "pred_risk": predictions["risk"],
        "plan": u_seq
    }

@nnx.jit
def _evaluate_sequences(z0, u_seq, r_fn, v_fn, f_fn, Q_fn, safety_threshold, discount=1.0, rho_u=0.0):
    return evaluate_sequences(z0, u_seq, r_fn, v_fn, f_fn, Q_fn, safety_threshold, discount, rho_u)