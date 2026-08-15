import jax
import jax.numpy as jnp
import flax.nnx as nnx
import cvxpy as cp
import numpy as np

from World_Model.Networks import RewardPredictor, DynamicsPredictor, ValueNet
from World_Model.Q_safety_critic import SafetyCriticEnsemble

@nnx.jit
def get_jacobians(
        z_ref: jax.Array, 
        u_ref: jax.Array, 
        r_fn: RewardPredictor, 
        v_fn: ValueNet, 
        f_fn: DynamicsPredictor, 
        Q_fn: SafetyCriticEnsemble,
        lambda_unc: float
    ):

    def reward_fn(z, u):
        return r_fn(z, u, update_spectral_norm=False)

    def value_fn(z):
        return v_fn(z, update_spectral_norm=False)

    def dyn_fn(z, u):
        return f_fn(z, u, update_spectral_norm=False)

    def safety_fn(z, u):
        return Q_fn.get_moments(z, u, update_spectral_norm=False)

    # Get reward jacobians
    r_jac_fn = jax.jacrev(reward_fn, argnums=(0, 1))
    Jr_z, Jr_u = jax.vmap(r_jac_fn, in_axes=(0, 0))(z_ref, u_ref)

    # Get value jacobians
    v_jac_fn = jax.jacrev(value_fn, argnums=(0,))
    Jv_z = jax.vmap(v_jac_fn, in_axes=(0,))(z_ref)

    # Get dynamics jacobians and dynamics prediction
    f_jac_fn = jax.jacfwd(dyn_fn, argnums=(0, 1))
    Jf_z, Jf_u = jax.vmap(f_jac_fn, in_axes=(0, 0))(z_ref, u_ref)
    f_ref = jax.vmap(dyn_fn, in_axes=(0, 0))

    # Get safety Jacobians and safety prediction
    Q_jac_fn = jax.jacrev(safety_fn, argnums=(0, 1))
    (JQmu_z, JQmu_u), (JQvar_z, JQvar_u) = jax.vmap(
        Q_jac_fn, in_axes=(0, 0)
    )(z_ref, u_ref)
    Q_mu, Q_var = jax.vmap(safety_fn, in_axes=(0, 0))

    # Get linearised dynamics system coefficients
    A = Jf_z
    B = Jf_u
    r = f_ref - (Jf_z @ z_ref) - (Jf_u @ u_ref)

    # Get safety system coefficients
    C = JQmu_z + lambda_unc * JQvar_z
    D = JQmu_u + lambda_unc * JQvar_u
    r_prime = (Q_mu - (JQmu_u @ u_ref) - (JQmu_z @ z_ref)) + lambda_unc * (
        Q_var - (JQvar_u @ u_ref) - (JQvar_z @ z_ref))
    
    return Jr_z, Jr_u, Jv_z, A, B, r, C, D, r_prime