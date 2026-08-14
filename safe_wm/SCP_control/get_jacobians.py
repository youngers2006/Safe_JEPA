import jax
import jax.numpy as jnp
import cvxpy as cp
import numpy as np

from World_Model.Networks import RewardPredictor, DynamicsPredictor, ValueNet
from World_Model.Q_safety_critic import SafetyCriticEnsemble

def get_jacobians(
        z_ref: jax.Array, 
        u_ref: jax.Array, 
        r_fn: RewardPredictor, 
        v_fn: ValueNet, 
        f_fn: DynamicsPredictor, 
        Q_fn: SafetyCriticEnsemble
    ):

    def reward_fn(z, u):
        return r_fn(z, u, update_spectral_norm=False)

    def value_fn(z):
        return v_fn(z, update_spectral_norm=False)

    def dyn_fn(z, u):
        return f_fn(z, u, update_spectral_norm=False)

    def safety_fn(z, u):
        return Q_fn.get_moments(z, u, update_spectral_norm=False)

    r_jac_fn = jax.jacfwd(reward_fn, argnums=(0, 1))
    Jr_z, Jr_u = jax.vmap(r_jac_fn, in_axes=(0, 0))(z_ref, u_ref)

    v_jac_fn = jax.jacfwd(value_fn, argnums=(0))
    Jv_z = jax.vmap(v_jac_fn, in_axes=(0))(z_ref)

    f_jac_fn = jax.jacfwd(dyn_fn, argnums=(0, 1))
    Jf_z, Jf_u = jax.vmap(f_jac_fn, in_axes=(0, 0))(z_ref, u_ref)

    Q_jac_fn = jax.jacfwd(safety_fn, argnums=(0, 1))
    (JQmu_z, JQmu_u), (JQvar_z, JQvar_u) = jax.vmap(
        Q_jac_fn, in_axes=(0, 0)
    )(z_ref, u_ref)
    
    return Jr_z, Jr_u, Jv_z, Jf_z, Jf_u, JQmu_z, JQmu_u, JQvar_z, JQvar_u