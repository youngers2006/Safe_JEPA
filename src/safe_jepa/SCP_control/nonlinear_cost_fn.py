import jax
import jax.numpy as jnp
import flax.nnx as nnx

from ..Sampling_control.evaluate_sequences import evaluate_sequences

@nnx.jit
def nonlinear_cost_fn(z0: jax.Array, u: jax.Array, r_fn, v_fn, f_fn, Q_fn, hyperparams: dict) -> tuple[jax.Array, jax.Array, dict]:
    # Extract hyperparams
    tau = hyperparams['tau']
    w_slack = hyperparams['w_slack']
    rho_u = hyperparams['rho_u']
    gamma = hyperparams['discount']

    _, feasible, _, predictions = evaluate_sequences(z0, u[None], r_fn, v_fn, f_fn, Q_fn, tau, gamma)
    predictions = {k: v[0] for k, v in predictions.items()}

    # Safety cost
    safe_errors = jnp.maximum(0.0, predictions["risk"] - tau)
    safe_cost = jnp.sum(safe_errors ** 2)

    # Dynamics cost
    dyn_cost = 0.0 # Must be 0 by definition here

    # Control cost
    control_cost = jnp.sum(u ** 2)

    J_nonlin = (-predictions["return"]
                + rho_u * control_cost
                + w_slack * (dyn_cost + safe_cost))

    # Also return the rollout so SCP can reuse it instead of rolling the plan out again
    return J_nonlin, feasible[0], predictions