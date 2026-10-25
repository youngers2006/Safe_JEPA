import jax
import jax.numpy as jnp
import flax.nnx as nnx

from ..Sampling_control.evaluate_sequences import evaluate_sequences

@nnx.jit
def nonlinear_cost_fn(z: jax.Array, u: jax.Array, r_fn, v_fn, f_fn, Q_fn, hyperparams: dict) -> tuple[jax.Array, jax.Array, dict]:
    # Extract hyperparams
    tau = hyperparams['tau']
    w_dyn = hyperparams['w_dyn']
    w_safe = hyperparams['w_safe']
    rho_u = hyperparams['rho_u']
    gamma = hyperparams['discount']
    H = u.shape[0]

    # Get predictions using model
    def reward_fn(z, u):
        return r_fn(z, u, update_spectral_norm=False).squeeze(-1)
 
    def dyn_fn(z, u):
        return f_fn(z, u, update_spectral_norm=False)
 
    def safety_fn(z, u):
        return Q_fn(z, u, update_spectral_norm=False).squeeze(axis=-1)
 
    rewards = jax.vmap(reward_fn)(z[:-1], u) # (H,)
    z_next = jax.vmap(dyn_fn)(z[:-1], u) # (H, d_z)
    risk = jax.vmap(safety_fn)(z[:-1], u) # (H,)
    terminal_value = v_fn(z[-1], update_spectral_norm=False).squeeze(-1) # ()

    # Compute predicted return
    discounted_return = jnp.sum(gamma ** jnp.arange(H) * rewards) + gamma ** H * terminal_value

    # Compute L1 penalty on defects, matches sum(mu_p + mu_n)
    error = z[1:] - z_next
    dyn_cost = jnp.sum(jnp.abs(error))

    # Compute L1 penalty on threshold violations, matches sum(nu)
    safe_cost = jnp.sum(jnp.maximum(0.0, risk - tau))

    # Compute control cost
    control_cost = jnp.sum(u ** 2)

    # Compute total cost and return it
    J_nonlinear = (-discounted_return
                   + rho_u * control_cost
                   + w_dyn * dyn_cost 
                   + w_safe * safe_cost)
    return J_nonlinear, jnp.max(jnp.abs(error)), jnp.max(risk)

@nnx.jit
def rollout_fn(z0: jax.Array, u: jax.Array, r_fn, v_fn, f_fn, Q_fn, hyperparams: dict) -> tuple[jax.Array, dict]:
    _, feasible, _, predictions = evaluate_sequences(
        z0, 
        u[None], 
        r_fn, 
        v_fn, 
        f_fn, 
        Q_fn, 
        hyperparams['tau'], 
        hyperparams['discount']
    )
    predictions = {k: v[0] for k, v in predictions.items()}
    return feasible[0], predictions