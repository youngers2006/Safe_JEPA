import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cp

from SCP_control.SCP_solver import ScpSolver
from SCP_control.get_jacobians import get_jacobians

class EconomicSCPSolver():
    def __init__(
            self, 
            jax_extractor_fn, # Jit compiled function
            SCP_solver: ScpSolver, 
            horizon: int, 
            d_z: int, 
            d_u: int, 
            hyperparams: dict, 
            scp_iters: int = 3, 
            tol: float = 1e-3
        ):
        self.jax_extractor_fn = jax_extractor_fn
        self.SCP_solver = SCP_solver
        self.horizon = horizon
        self.hyperparams = hyperparams
        self.scp_iters = scp_iters
        self.tol = tol
        
        # Memory buffer for trajectories to use in warm starting
        self.u_prev = np.zeros((horizon, d_u))
        self.z_prev = np.zeros((horizon + 1, d_z))

    def step(self, z_c, wm_networks):
        # Extract networks
        r_fn, v_fn, f_fn, Q_fn = wm_networks

        # Shift actions by 1 and set last action to 0
        u_ref = np.roll(self.u_prev, shift=-1, axis=0)
        u_ref[-1, :] = 0.0

        # Shift states forward by 1 and set last state to the same as previous
        z_ref = np.roll(self.z_prev, shift=-1, axis=0)
        z_ref[-1, :] = z_ref[-2, :]

        # Anchor states to observation
        z_ref[0, :] = z_c

        for _ in range(self.scp_iters):
            # Extract matrices from world model
            jax_matrices = self.jax_extractor_fn(
                z_ref=jnp.array(z_ref), 
                u_ref=jnp.array(u_ref),
                r_fn=r_fn,
                v_fn=v_fn,
                f_fn=f_fn,
                Q_fn=Q_fn,
                lambda_unc=self.hyperparams["lambda_unc"]
            )

            # Run solver to setup and solve SOCP
            u_opt, z_opt = self.SCP_solver.solve_problem(
                jax_matrices,
                z_c,
                z_ref,
                u_ref,
                self.hyperparams
            )

            # Calculate change in u
            delta_u = np.max(np.abs(u_opt - u_ref))

            # If solution has converged, break the loop
            if delta_u < self.tol:
                break

            # Update reference
            u_ref = u_opt
            z_ref = z_opt

        # Update orevious solution buffer and return 1st action to take
        self.u_prev = u_opt
        self.z_prev = z_opt
        return u_opt[0, :]