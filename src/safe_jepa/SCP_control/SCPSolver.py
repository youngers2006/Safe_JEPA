import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cp

class SCPSolver():
    def __init__(self, horizon: int, d_z: int, d_u: int, discount: float, u_min: float = -1.0, u_max: float = 1.0):
        # Constants
        self.horizon = horizon
        self.d_z = d_z
        self.d_u = d_u
        self.discount = discount

        # Optimisation variables (real and slack)
        self.z = cp.Variable((horizon + 1, d_z))
        self.u = cp.Variable((horizon, d_u))
        self.mu_p = cp.Variable((horizon, d_z), nonneg=True)
        self.mu_n = cp.Variable((horizon, d_z), nonneg=True)
        self.nu = cp.Variable(horizon, nonneg=True)

        # Reference path scaled by the trust region weight, w_prox * z_ref and w_prox * u_ref: expanding the proximal term
        # w_prox ||z - z_ref||^2 = w_prox ||z||^2 - 2 (w_prox z_ref)^T z + const keeps the problem DPP without auxiliary
        # deviation variables (w_prox * sum_squares(z - z_ref) would multiply two parameter-dependent expressions)
        self.wz_ref = cp.Parameter((horizon + 1, d_z))
        self.wu_ref = cp.Parameter((horizon, d_u))

        # Taylor series coefficients
        self.A = [cp.Parameter((d_z, d_z)) for _ in range(self.horizon)]
        self.B = [cp.Parameter((d_z, d_u)) for _ in range(self.horizon)]
        self.r = [cp.Parameter(d_z) for _ in range(self.horizon)]
        self.C = [cp.Parameter(d_z) for _ in range(self.horizon)]
        self.D = [cp.Parameter(d_u) for _ in range(self.horizon)]
        self.r_prime = [cp.Parameter() for _ in range(self.horizon)]
        self.Jr_z = [cp.Parameter(d_z) for _ in range(self.horizon)]
        self.Jr_u = [cp.Parameter(d_u) for _ in range(self.horizon)]
        self.Jv_z = cp.Parameter(d_z)
        self.z_c = cp.Parameter(d_z)

        # Safety probability threshold
        self.tau = cp.Parameter(nonneg=True)

        # Weights for optimisation
        self.rho_u = cp.Parameter(nonneg=True)
        self.w_slack = cp.Parameter(nonneg=True)
        self.w_prox = cp.Parameter(nonneg=True)

        # Initialise problem store
        constraints = [self.z[0, :] == self.z_c]
        objective_terms = []

        for h in range(self.horizon):
            # Dynamics constraint: z_k+1 = A_k z_k + B_k u_k + r_k + mu_p - mu_n
            constraints.append(
                self.z[h+1] == self.A[h] @ self.z[h] + self.B[h] @ self.u[h] + self.r[h] + self.mu_p[h] - self.mu_n[h]
            )

            # Safety constraint: C_k z_k + D_k u_k - nu_k <= tau - r'_k
            constraints.append(self.C[h] @ self.z[h] + self.D[h] @ self.u[h] - self.nu[h] <= 
                self.tau - self.r_prime[h])

            # Action bounding
            constraints.append(self.u[h] <= u_max)
            constraints.append(self.u[h] >= u_min)

            # Timestep cost (reward, slack variable costs, trust region terms on z_1...z_H and u_0...u_H-1, and the action
            # magnitude cost, which shares the quadratic in u with the trust region term)
            objective_terms.append(-(self.discount ** h) * (self.Jr_z[h] @ self.z[h] + self.Jr_u[h] @ self.u[h])
                    + self.w_slack * cp.sum(self.mu_p[h] + self.mu_n[h]) + self.w_slack * cp.sum(self.nu[h])
                    + self.w_prox * cp.sum_squares(self.z[h+1]) - 2.0 * (self.wz_ref[h+1] @ self.z[h+1])
                    + (self.w_prox + self.rho_u) * cp.sum_squares(self.u[h]) - 2.0 * (self.wu_ref[h] @ self.u[h]))

        # End of horizon cost with value function
        objective_terms.append(-(self.discount ** self.horizon) * (self.Jv_z @ self.z[self.horizon]))

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
        self.z_c.value = z_c
        self.z_ref_np, self.u_ref_np = np.asarray(z_ref), np.asarray(u_ref)
        self.wz_ref.value = float(hyperparams['w_prox']) * self.z_ref_np
        self.wu_ref.value = float(hyperparams['w_prox']) * self.u_ref_np
        self.Jv_z.value = Jv_z_np

        # Setup hyperparams
        self.tau.value = float(hyperparams['tau'])
        self.rho_u.value = float(hyperparams['rho_u'])
        self.w_slack.value = float(hyperparams['w_slack'])
        self.w_prox.value = float(hyperparams['w_prox'])

        # Setup time varying values
        for h in range(self.horizon):
            self.A[h].value = A_np[h]
            self.B[h].value = B_np[h]
            self.r[h].value = r_np[h]
            self.C[h].value = C_np[h]
            self.D[h].value = D_np[h]
            self.r_prime[h].value = r_prime_np[h]
            self.Jr_z[h].value = Jr_z_np[h]
            self.Jr_u[h].value = Jr_u_np[h]

        try:
            # Solve the QP with OSQP, warm started from the last solution; polishing recovers the exact active-set solution
            self.problem.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-5, eps_rel=1e-5, max_iter=20000, polish=True)
        except Exception as e:
            print(f"SCP subproblem: solver crashed ({e})")
            return self.u_ref_np, self.z_ref_np, False
        
        status = self.problem.status
        if status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and self.u.value is not None: # Success
            return self.u.value, self.z.value, True
        print(f"SCP subproblem: solver returned status {status}")
        return self.u_ref_np, self.z_ref_np, False