import jax.numpy as jnp
import cvxpy as cp
import numpy as np

def get_jacobians(z, u, r_fn, v_fn, f_fn, Q_fn):
    return dr_dz, dr_du, dv_dz, df_dz, df_du, dQmu_dz, dQmu_du, dQvar_dz, dQvar_du