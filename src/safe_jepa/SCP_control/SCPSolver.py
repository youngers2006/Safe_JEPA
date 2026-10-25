import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cp

class SCPSolver():
    def __init__(self, horizon: int, d_z: int, d_u: int, discount: float, rho_u: float, u_min: float = -1.0, u_max: float = 1.0):
        # Constants
        self.horizon = horizon
        self.d_z = d_z
        self.d_u = d_u
        self.discount = discount
        self.rho_u = rho_u
 
        # Optimisation variables: deviations from the reference trajectory, and slacks
        # dz[h] = z_{h+1} - z_ref_{h+1} (covers z_1..z_H, z_0 is fixed), du[h] = u_h - u_ref_h
        self.dz = cp.Variable((horizon, d_z))
        self.du = cp.Variable((horizon, d_u))
        self.mu_p = cp.Variable((horizon, d_z), nonneg=True)
        self.mu_n = cp.Variable((horizon, d_z), nonneg=True)
        self.nu = cp.Variable(horizon, nonneg=True)
 
        # Reference path
        self.z_ref = cp.Parameter((horizon + 1, d_z))
        self.u_ref = cp.Parameter((horizon, d_u))
 
        # Taylor series coefficients, expanded around the reference
        self.A = [cp.Parameter((d_z, d_z)) for _ in range(self.horizon)]
        self.B = [cp.Parameter((d_z, d_u)) for _ in range(self.horizon)]
        self.f = [cp.Parameter(d_z) for _ in range(self.horizon)]
        self.C = [cp.Parameter(d_z) for _ in range(self.horizon)]
        self.D = [cp.Parameter(d_u) for _ in range(self.horizon)]
        self.q = [cp.Parameter() for _ in range(self.horizon)]
        self.Jr_z = [cp.Parameter(d_z) for _ in range(self.horizon)]
        self.Jr_u = [cp.Parameter(d_u) for _ in range(self.horizon)]
        self.Jv_z = cp.Parameter(d_z)
 
        # Safety probability threshold
        self.tau = cp.Parameter(nonneg=True)
 
        # Weights for optimisation
        self.w_slack = cp.Parameter(nonneg=True)
        self.w_prox = cp.Parameter(nonneg=True)
 
        # Initialise problem store
        constraints = []
        objective_terms = []

        for h in range(self.horizon):
            dyn_rhs = self.f[h] + self.B[h] @ self.du[h] + self.mu_p[h] - self.mu_n[h]
            if h > 0:
                dyn_rhs = dyn_rhs + self.A[h] @ self.dz[h-1]
            constraints.append(self.z_ref[h+1] + self.dz[h] == dyn_rhs)
 
            # Linearised safety around the reference: Q_ref_h + C_h dz_{h-1} + D_h du_h - nu_h <= tau
            safe_lhs = self.q[h] + self.D[h] @ self.du[h] - self.nu[h]
            if h > 0:
                safe_lhs = safe_lhs + self.C[h] @ self.dz[h - 1]
            constraints.append(safe_lhs <= self.tau)
 
            # Action bounding
            constraints.append(self.u_ref[h] + self.du[h] <= u_max)
            constraints.append(self.u_ref[h] + self.du[h] >= u_min)
 
            # Timestep cost (action magnitude cost, reward, slack variable costs, trust region costs)
            reward = self.Jr_u[h] @ self.du[h]
            if h > 0:
                reward = reward + self.Jr_z[h] @ self.dz[h - 1]
            objective_terms.append(
                self.rho_u * cp.sum_squares(self.u_ref[h] + self.du[h])
                - (self.discount ** h) * reward
                + self.w_slack * cp.sum(self.mu_p[h] + self.mu_n[h]) + self.w_slack * self.nu[h]
                + self.w_prox * (cp.sum_squares(self.dz[h]) + cp.sum_squares(self.du[h]))
            )
        objective_terms.append(-(self.discount ** self.horizon) * (self.Jv_z @ self.dz[self.horizon - 1]))
 
        # Setup problem with cvxpy
        self.problem = cp.Problem(cp.Minimize(cp.sum(objective_terms)), constraints)
 
        # Ensure problem is DPP
        assert self.problem.is_dpp(), "Warning: SCP problem is not DPP"

    def solve_problem(
            self, 
            system_matrices, 
            z_c, 
            z_ref, 
            u_ref, 
            hyperparams: dict
        ):
        # Extract matrices from tuple
        Jr_z_np, Jr_u_np, Jv_z_np, A_np, B_np, r_np, C_np, D_np, r_prime_np = system_matrices

        # Setup static values
        self.z_ref.value = z_ref
        self.u_ref.value = u_ref
        self.Jv_z.value = Jv_z_np

        # Setup hyperparams
        self.tau.value = float(hyperparams['tau'])
        self.w_slack.value = float(hyperparams['w_slack'])
        self.w_prox.value = float(hyperparams['w_prox'])

        f_np = (A_np @ z_ref[:-1, :, np.newaxis]).squeeze(-1) + (B_np @ u_ref[:, :, np.newaxis]).squeeze(-1) + r_np
        q_np = ((C_np[:, np.newaxis, :] @ z_ref[:-1, :, np.newaxis]).squeeze((-2, -1))
        + (D_np[:, np.newaxis, :] @ u_ref[:, :, np.newaxis]).squeeze((-2, -1))
        + r_prime_np)

        # Setup time varying values
        for h in range(self.horizon):
            self.A[h].value = A_np[h]
            self.B[h].value = B_np[h]
            self.f[h].value = f_np[h]
            self.C[h].value = C_np[h]
            self.D[h].value = D_np[h]
            self.q[h].value = q_np[h]
            self.Jr_z[h].value = Jr_z_np[h]
            self.Jr_u[h].value = Jr_u_np[h]

        try:
            # Solve optimisation problem with SOCP solver CLARABEL
            self.problem.solve(solver=cp.CLARABEL, warm_start=True)
        except Exception as e:
            print(f"SCP subproblem: solver crashed ({e})")
            return self.u_ref.value, self.z_ref.value, False
        
        status = self.problem.status
        if status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and self.du.value is not None: # Success
            u = u_ref + self.du.value
            z = np.vstack([np.asarray(z_c)[np.newaxis], z_ref[1:] + self.dz.value])
            return u, z, True
        print(f"SCP subproblem: solver returned status {status}")
        return self.u_ref.value, self.z_ref.value, False