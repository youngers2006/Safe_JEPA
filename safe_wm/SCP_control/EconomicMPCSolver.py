import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cp

from SCP_control.SCP_solver import ScpSolver
from SCP_control.get_jacobians import get_jacobians
from SCP_control.nonlinear_cost_fn import nonlinear_cost_fn

class EconomicSCPSolver():
    def __init__(
            self, 
            jax_extractor_fn, # Jit compiled function
            non_linear_cost_fn, # Jit compiled function
            SCP_solver: ScpSolver, 
            horizon: int, 
            d_z: int, 
            d_u: int, 
            hyperparams: dict, 
            scp_iters: int = 3, 
            rho_lower: float = 0.25,
            rho_upper: float = 0.75,
            tol: float = 1e-3
        ):
        self.jax_extractor_fn = jax_extractor_fn
        self.SCP_solver = SCP_solver
        self.horizon = horizon
        self.hyperparams = hyperparams
        self.scp_iters = scp_iters
        self.tol = tol

        self.rho_lower = rho_lower
        self.rho_upper = rho_upper
        
        # Memory buffer for trajectories to use in warm starting
        self.u_prev = np.zeros((horizon, d_u))
        self.z_prev = np.zeros((horizon + 1, d_z))

    def linear_cost_fn(self, system_matrices: tuple[np.ndarray, ...], z: np.ndarray, u: np.ndarray) -> np.ndarray:
        # Extract system matrices
        Jr_z, Jr_u, Jv_z, A, B, r, C, D, r_prime = system_matrices

        # Hyperparams
        gamma = self.hyperparams['discount']
        rho_u = self.hyperparams['rho_u']
        w_slack = self.hyperparams['w_slack']
        tau = self.hyperparams['tau']
        
        # Vectorized discount array [1, gamma, gamma^2, ...]
        discount_vec = gamma ** np.arange(self.horizon)

        # Define unsqeezed vectors to allow @ operation on 3rd order tensor
        z_un = z[..., np.newaxis]
        u_un = u[..., np.newaxis]

        # Reward for trajectory
        reward_vec = np.sum(Jr_z * z[:-1], axis=1) + np.sum(Jr_u * u, axis=1)
        discounted_reward = np.sum(discount_vec * reward_vec)

        # mu_p and mu_n usage penalty
        Az = (A @ z_un[:-1]).squeeze(-1)
        Bu = (B @ u_un).squeeze(-1)
        dyn_error = z[1:] - (Az + Bu + r)
        dyn_penalty = np.sum(dyn_error ** 2)

        # nu usage penalty
        Cz = np.sum(C * z[:-1], axis=1)
        Du = np.sum(D * u, axis=1)
        safe_pred = Cz + Du + r_prime
        safe_error = np.maximum(0.0, safe_pred - tau)
        safe_penalty = np.sum(safe_error ** 2)

        # Action magnitude penalty
        control_penalty = rho_u * np.sum(u ** 2)

        # Terminal value reward
        terminal_value = (gamma ** self.horizon) * np.sum(Jv_z * z[-1]) 

        # Compute total trajectory cost for the linearised system
        J_lin = (-discounted_reward 
                 + control_penalty 
                 + w_slack * (dyn_penalty + safe_penalty) 
                 - terminal_value)
        return J_lin

    def non_linear_cost_fn(self, wm_networks: tuple[jax.Array, ...], z_c: jax.Array, u: jax.Array) -> np.ndarray:
        cost_jax = self.nonlinear_cost_fn(
            jnp.array(z_c), 
            jnp.array(u),
            wm_networks,
            self.hyperparams
        )
        return np.asarray(cost_jax)

    def step(self, z_c: jax.Array, wm_networks: tuple[jax.Array, ...]) -> jax.Array:
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

            # Conversion to numpy
            cpu_matrices = tuple(np.asarray(m) for m in jax_matrices)
            z_c_np = np.asarray(z_c)
            z_ref_np = np.asarray(z_ref)
            u_ref_np = np.asarray(u_ref)

            # Run solver to setup and solve SOCP
            u_opt, z_opt = self.SCP_solver.solve_problem(
                cpu_matrices,
                z_c_np,
                z_ref_np,
                u_ref_np,
                self.hyperparams
            )

            # Compute reference trajectory costs with the model and linearised system
            linear_cost_old = self.linear_cost_fn(
                cpu_matrices, z_ref_np, u_ref_np
            )
            nonlinear_cost_old = self.non_linear_cost_fn(
                wm_networks, z_c, u_ref
            )

            # Compute optimal trajectory costs with model and linearised system
            linear_cost_new = self.linear_cost_fn(
                cpu_matrices, z_ref_np, u_opt
            )
            nonlinear_cost_new = self.non_linear_cost_fn(
                wm_networks, z_c, u_opt
            )

            # Compute the ratio of cost improvement between true and linearised system
            dJ_actual = nonlinear_cost_old - nonlinear_cost_new
            dJ_predicted = linear_cost_old - linear_cost_new
            rho = dJ_actual / dJ_predicted

            if rho < self.rho_lower: # Reject u* and decrease TR size
                self.hyperparams['w_prox'] = 2.0 * self.hyperparams['w_prox']
            else:
                # Calculate change in u
                delta_u = np.max(np.abs(u_opt - u_ref))

                # Update reference
                u_ref = u_opt
                z_ref = z_opt

                # Check if TR must be resized
                if rho > self.rho_upper:
                    self.hyperparams['w_prox'] = 0.5 * self.hyperparams['w_prox']
                else:
                    pass

                # If solution has converged, break the loop
                if delta_u < self.tol:
                    break

        # Update orevious solution buffer and return 1st action to take
        self.u_prev = u_opt
        self.z_prev = z_opt
        return jnp.array(u_opt[0, :])