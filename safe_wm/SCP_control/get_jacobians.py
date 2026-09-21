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
        return r_fn(z, u, update_spectral_norm=False).squeeze(-1)

    def value_fn(z):
        return v_fn(z, update_spectral_norm=False).squeeze(-1)

    def dyn_fn(z, u):
        return f_fn(z, u, update_spectral_norm=False)

    def safety_fn(z, u):
        mu, std_d = Q_fn.get_moments(z, u, update_spectral_norm=False)
        return mu.squeeze(-1), std_d.squeeze(-1)

    # Get reward jacobians
    r_jac_fn = jax.jacrev(reward_fn, argnums=(0, 1))
    Jr_z, Jr_u = jax.vmap(r_jac_fn, in_axes=(0, 0))(z_ref[:-1], u_ref)

    # Get value jacobian
    v_jac_fn = jax.jacrev(value_fn)
    Jv_z = v_jac_fn(z_ref[-1])

    # Get dynamics jacobians and dynamics prediction
    f_jac_fn = jax.jacfwd(dyn_fn, argnums=(0, 1))
    Jf_z, Jf_u = jax.vmap(f_jac_fn, in_axes=(0, 0))(z_ref[:-1], u_ref)
    f_ref = jax.vmap(dyn_fn, in_axes=(0, 0))(z_ref[:-1], u_ref)

    # Get safety Jacobians and safety prediction
    Q_jac_fn = jax.jacrev(safety_fn, argnums=(0, 1))
    (JQmu_z, JQmu_u), (JQstd_d_z, JQstd_d_u) = jax.vmap(
        Q_jac_fn, in_axes=(0, 0)
    )(z_ref[:-1], u_ref)
    Q_mu, Q_std_d = jax.vmap(safety_fn, in_axes=(0, 0))(z_ref[:-1], u_ref)

    # Get linearised dynamics system coefficients
    z_un = z_ref[:-1, ..., jnp.newaxis]
    u_un = u_ref[..., jnp.newaxis]
    A = Jf_z
    B = Jf_u
    r = f_ref - (Jf_z @ z_un).squeeze(-1) - (Jf_u @ u_un).squeeze(-1)

    # Get safety system coefficients
    C = JQmu_z + lambda_unc * JQstd_d_z
    D = JQmu_u + lambda_unc * JQstd_d_u
    r_prime = (
        Q_mu 
        - jnp.sum(JQmu_u * u_ref, axis=-1) 
        - jnp.sum(JQmu_z * z_ref[:-1], axis=-1)
    ) + lambda_unc * (
        Q_std_d 
        - jnp.sum(JQstd_d_u * u_ref, axis=-1) 
        - jnp.sum(JQstd_d_z * z_ref[:-1], axis=-1)
    )

    # Check for debugging
    N = u_ref.shape[0]
    assert Jr_z.shape == (N, z_ref.shape[-1])
    assert Jv_z.shape == (z_ref.shape[-1],)
    assert A.shape == (N, z_ref.shape[-1], z_ref.shape[-1])
    assert C.shape == (N, z_ref.shape[-1]) and D.shape == (N, u_ref.shape[-1])
    assert r_prime.shape == (N,)
    
    return Jr_z, Jr_u, Jv_z, A, B, r, C, D, r_prime