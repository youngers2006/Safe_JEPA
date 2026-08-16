import jax
import jax.numpy as jnp
import flax.nnx as nnx

@nnx.jit
def nonlinear_cost_fn(z_c, u, r_fn, v_fn, f_fn, Q_fn, hyperparams):
    # Get computation graphs of the networks for lax scan
    r_graph, r_state = nnx.split(r_fn)
    f_graph, f_state = nnx.split(f_fn)
    Q_graph, Q_state = nnx.split(Q_fn)

    # extract hyperparams
    tau = hyperparams['tau']
    w_slack = hyperparams['w_slack']
    rho_u = hyperparams['rho_u']
    gamma = hyperparams['discount']
    lambda_unc = hyperparams['lambda_unc']

    def scan_step(z_current, u_current):
        z_next = f_graph.apply(f_state)(z_current, u_current, update_spectral_norm=False)
        reward = r_graph.apply(r_state)(z_current, u_current, update_spectral_norm=False)
        q_mu, q_var = Q_graph.apply(Q_state).get_moments(z_current, u_current, update_spectral_norm=False)
        q_safe = q_mu + lambda_unc * q_var
        return z_next, (z_next, reward, q_safe) # Returns Carry variable (z_next) and yield variables (z_next, reward, q_mu, q_var)

    # Scan step function to run it in compiled form over a loop up to the horizon
    z_terminal, (z_history, r_history, q_safe_history) = jax.lax.scan(scan_step, z_c, u)

    # Compute discount vector
    horizon = u.shape[0]
    discount_vec = gamma ** jnp.arange(horizon)

    # Reward cost
    discounted_reward = jnp.sum(discount_vec * r_history)

    # Safety cost
    safe_errors = jnp.maximum(0.0, q_safe_history - tau)
    safe_cost = jnp.sum(safe_errors ** 2)

    # Dynamics cost
    dyn_cost = 0.0 # Must be 0 by definition here

    # Control cost
    control_cost = jnp.sum(u ** 2)

    # Terminal value cost
    v_terminal = v_fn(z_terminal, update_spectral_norm=False)
    terminal_term = (gamma ** horizon) * v_terminal
    
    # Compute final penalty
    J_nonlin = (-discounted_reward 
                + rho_u * control_cost 
                + w_slack * (dyn_cost + safe_cost) 
                - terminal_term)
                
    return J_nonlin