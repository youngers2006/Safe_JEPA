import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cp

class ScpSolver():
    def __init__(self, horizon: int, d_z: int, d_u: int, u_min:float = -1.0, u_max: float = 1.0):
        # Constants
        self.horizon = horizon
        self.d_z = d_z
        self.d_u = d_u

        # Optimisation variables (real and slack)
        self.z = cp.Variable((horizon + 1, d_z))
        self.u = cp.Variable((horizon, d_u))
        self.t = cp.Variable((horizon))
        self.mu_p = cp.Variable((horizon, d_z), nonneg=True)
        self.mu_n = cp.Variable((horizon, d_z), nonneg=True)
        self.nu = cp.Variable(horizon, nonneg=True)

        # Reference path
        self.z_ref = cp.Parameter((horizon + 1, d_z))
        self.u_ref = cp.Parameter((horizon, d_u))
        self.t_ref = cp.Parameter((horizon))

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
        self.discount = cp.Parameter(nonneg=True)

        # Initialise problem store
        constraints = [self.z[0, :] == self.z_c]
        objective_terms = []

        for h in range(self.horizon):
            # Dynamics constraint: z_k+1 = A_k z_k + B_k u_k + r_k + mu_p - mu_n
            constraints.append(self.z[h+1] == 
                self.A[h] @ self.z[h] + self.B[h] @ self.u[h] + self.r[h] + self.mu_p[h] - self.mu_n[h])

            # Safety constraint: C_k z_k + D_k u_k - nu_k <= tau - r'_k
            constraints.append(self.C[h] @ self.z[h] + self.D[h] @ self.u[h] - self.nu[h] <= 
                self.tau - self.r_prime[h])

            # Action bounding
            constraints.append(self.u[h] <= u_max) ; constraints.append(self.u[h] >= u_min)

            # Epigraph condition: moved action magnitude cost to conditions with epigraph to allow otpimisation
            constraints.append(self.rho_u * cp.sum_squares(self.u[h]) - self.t[h] <= 0)

            # Timestep cost (action magnitude cost, reward, slack variable costs, trust region costs)
            objective_terms.append(self.t[h] - (self.discount ** h) * (self.Jr_z[h] @ self.z[h] + self.Jr_u[h] @ self.u[h])
                        + self.w_slack * cp.sum_squares(self.mu_p[h] + self.mu_n[h]) + self.w_slack * cp.sum_squares(self.nu[h]) +
                        self.w_prox * (cp.sum_squares(self.z[h] - self.z_ref[h]) + cp.sum_squares(self.u[h] - self.u_ref[h])))

        # End of horizon cost with value function
        objective_terms.append(-(self.discount ** (self.horizon)) * (self.Jv_z @ self.z[self.horizon]))

        # Setup problem with cvxpy
        self.problem = cp.Problem(cp.Minimize(cp.sum(objective_terms)), constraints)

    def solve_problem(
            self, 
            system_matrices: tuple[jax.Array, ...], 
            z_c: jax.Array, 
            z_ref: jax.Array, 
            u_ref: jax.Array, 
            hyperparams: dict
        ):
        # Extract matrices from tuple
        Jr_z_jax, Jr_u_jax, Jv_z_jax, A_jax, B_jax, r_jax, C_jax, D_jax, r_prime_jax = system_matrices

        # Move all matrices onto cpu and into numpy
        A_np = np.asarray(A_jax)
        B_np = np.asarray(B_jax)
        r_np = np.asarray(r_jax)
        C_np = np.asarray(C_jax)
        D_np = np.asarray(D_jax)
        r_prime_np = np.asarray(r_prime_jax)
        Jr_z_np = np.asarray(Jr_z_jax)
        Jr_u_np = np.asarray(Jr_u_jax)

        # Setup static values
        self.z_c.value = np.asarray(z_c)
        self.z_ref.value = np.asarray(z_ref)
        self.u_ref.value = np.asarray(u_ref)
        self.Jv_z.value = np.asarray(Jv_z_jax)

        # Setup hyperparams
        self.tau.value = float(hyperparams['tau'])
        self.rho_u.value = float(hyperparams['rho_u'])
        self.w_slack.value = float(hyperparams['w_slack'])
        self.w_prox.value = float(hyperparams['w_prox'])
        self.discount.value = float(hyperparams['discount'])

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
            self.problem.solve(solver=cp.CLARABEL, warm_start=True)
            status = self.problem.status

            if status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]: # Success
                return self.u.value, self.z.value
            elif status == cp.INFEASIBLE: # Failed due to infeasible preoblem
                print("SCP Error: Problem is Infeasible. Falling Back to Previous Solution")
                return self.z_ref.value, self.u_ref.value
            else: # Failed for other reason
                print(f"Solver Error: Solver exited with status {status}")
                return self.z_ref.value, self.u_ref.value
            
        except cp.SolverError as e:
            print(f"C++ Backend Crashed: {e}")