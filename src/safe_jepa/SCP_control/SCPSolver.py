import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import clarabel
import scipy.sparse as sp

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
        self.w_dyn = cp.Parameter(nonneg=True)
        self.w_safe = cp.Parameter(nonneg=True)
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
                + self.w_dyn * cp.sum(self.mu_p[h] + self.mu_n[h]) + self.w_safe * self.nu[h]
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
        self.w_dyn.value = float(hyperparams['w_dyn'])
        self.w_safe.value = float(hyperparams['w_safe'])
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

class DirectSCPSolver():
    def __init__(
            self, 
            horizon: int, 
            d_z: int, 
            d_u: int, 
            discount: float, 
            rho_u: float, 
            u_min: float = -1.0, 
            u_max: float = 1.0
        ):
        # Constants
        self.horizon = horizon
        self.d_z = d_z
        self.d_u = d_u
        self.discount = discount
        self.rho_u = rho_u

        # Setup of the problem
        H = horizon
        self.n_z, self.n_u, self.n_mu = (H + 1) * d_z, H * d_u, H * d_z # dimensions
        self.o_u, self.o_mp = self.n_z, self.n_z + self.n_u # offset
        self.o_mn, self.o_nu = self.o_mp + self.n_mu, self.o_mp + 2 * self.n_mu # offset
        self.n = self.o_nu + H # total number of unknowns
        z_col = lambda h, i: h * d_z + i
        u_col = lambda h, j: self.o_u + h * d_u + j
        r_dyn, r_safe = d_z, d_z + H * d_z
        r_up = r_safe + H
        r_lo = r_up + self.n_u
        r_slack = r_lo + self.n_u
        self.m = r_slack + 2 * self.n_mu + H

        # Setup matrices
        rows, cols = [], []
        for h in range(H): # A_h on z_h
            for i in range(d_z):
                rows += [r_dyn + h * d_z + i] * d_z
                cols += [z_col(h,j) for j in range(d_z)]
        for h in range(H): # B_h on u_h
            for i in range(d_z):
                rows += [r_dyn + h * d_z + i] * d_u; cols += [u_col(h, j) for j in range(d_u)]
        for h in range(H): # C_h on z_h
            rows += [r_safe + h] * d_z; cols += [z_col(h, j) for j in range(d_z)]
        for h in range(H): # D_h on u_h
            rows += [r_safe + h] * d_u; cols += [u_col(h, j) for j in range(d_u)]

        const = [] # the entries that never change
        for i in range(d_z): # z_0 = z_c
            rows.append(i); cols.append(z_col(0, i)); const.append(1.0)
        for h in range(H):
            for i in range(d_z): # + z_h+1 - mu_p + mu_n in the dynamics rows
                row = r_dyn + h * d_z + i
                rows += [row] * 3; cols += [z_col(h + 1, i), self.o_mp + h * d_z + i, self.o_mn + h * d_z + i]
                const += [1.0, -1.0, 1.0]
            rows.append(r_safe + h); cols.append(self.o_nu + h); const.append(-1.0) # - nu in the safety rows
        for k in range(self.n_u): # action bounds
            rows += [r_up + k, r_lo + k]; cols += [self.o_u + k] * 2; const += [1.0, -1.0]
        for k in range(2 * self.n_mu + H): # slacks >= 0
            rows.append(r_slack + k); cols.append(self.o_mp + k); const.append(-1.0)
        self.const_vals = np.array(const)

        # Clarabel takes A in compressed-column form; record where each entry listed above lands in its data array
        order = sp.csc_matrix((np.arange(len(rows), dtype=float) + 1.0, (rows, cols)), shape=(self.m, self.n))
        self.A_perm = order.data.astype(int) - 1
        self.A_indices, self.A_indptr = order.indices, order.indptr

        # Right-hand sides that never change (action bounds, slacks >= 0) and the cones: equalities, then inequalities
        self.b = np.zeros(self.m)
        self.b[r_up:r_lo], self.b[r_lo:r_slack] = u_max, -u_min
        self.rows = (r_dyn, r_safe, r_up)
        self.cones = [clarabel.ZeroConeT(r_safe), clarabel.NonnegativeConeT(self.m - r_safe)]
        self.settings = clarabel.DefaultSettings()
        self.settings.verbose = False
        self.settings.presolve_enable = False # presolve would stop the problem data being updated in place

        # P is diagonal: trust region on z_1..z_H, trust region plus action cost on u
        self.P_idx = np.arange(d_z, self.o_mp)
        self.solver, self.result = None, None
        
    def slack_and_multipliers(self) -> tuple[float, float, float, float]:
        # Largest dynamics and safety slack at the last solution, and the largest multiplier of each. A multiplier at
        # its L1 weight means that slack is in use, so w_dyn and w_safe must sit above the multipliers
        x, y = np.asarray(self.result.x), np.asarray(self.result.z)
        r_dyn, r_safe, r_up = self.rows
        return (float(np.max(x[self.o_mp:self.o_mn] + x[self.o_mn:self.o_nu])), float(np.max(x[self.o_nu:])),
                float(np.max(np.abs(y[r_dyn:r_safe]))), float(np.max(y[r_safe:r_up])))

    def solve_problem(
            self,
            system_matrices,
            z_c,
            z_ref,
            u_ref,
            hyperparams: dict
        ):
        # Extract matrices from tuple
        Jr_z, Jr_u, Jv_z, A, B, r, C, D, r_prime = system_matrices
        H, d_z = self.horizon, self.d_z
        w_prox = float(hyperparams['w_prox'])
        r_dyn, r_safe, r_up = self.rows

        # Constraint matrix values, in compressed-column order
        vals = np.concatenate([-np.asarray(A).ravel(), -np.asarray(B).ravel(), np.asarray(C).ravel(),
                               np.asarray(D).ravel(), self.const_vals])
        A_data = vals[self.A_perm]

        # Right-hand sides: z_0 = z_c, the dynamics' constant terms, safety below tau - r'
        b = self.b.copy()
        b[:d_z] = z_c
        b[r_dyn:r_safe] = np.asarray(r).ravel()
        b[r_safe:r_up] = float(hyperparams['tau']) - np.asarray(r_prime)

        # Linear terms: rewards and terminal value, the trust region's centre, the slack weights
        disc = self.discount ** np.arange(H)
        q = np.zeros(self.n)
        q[:H * d_z] -= (disc[:, None] * Jr_z).ravel()
        q[H * d_z:self.n_z] -= (self.discount ** H) * np.asarray(Jv_z)
        q[d_z:self.n_z] -= 2.0 * w_prox * np.asarray(z_ref)[1:].ravel()
        q[self.o_u:self.o_mp] = -(disc[:, None] * Jr_u).ravel() - 2.0 * w_prox * np.asarray(u_ref).ravel()
        q[self.o_mp:self.o_nu] = float(hyperparams['w_dyn'])
        q[self.o_nu:] = float(hyperparams['w_safe'])

        # Quadratic terms (1/2 x'Px, so twice the weights)
        P_data = np.concatenate([np.full(self.n_z - d_z, 2.0 * w_prox), np.full(self.n_u, 2.0 * (w_prox + self.rho_u))])

        try:
            # Update the numbers in place where this Clarabel version allows it, otherwise build the solver again
            if self.solver is not None and getattr(self.solver, "is_data_update_allowed", lambda: False)():
                self.solver.update(P=P_data, q=q, A=A_data, b=b)
            else:
                P = sp.csc_matrix((P_data, (self.P_idx, self.P_idx)), shape=(self.n, self.n))
                A_mat = sp.csc_matrix((A_data, self.A_indices, self.A_indptr), shape=(self.m, self.n))
                self.solver = clarabel.DefaultSolver(P, q, A_mat, b, self.cones, self.settings)
            res = self.solver.solve()
        except Exception as e:
            print(f"SCP subproblem: solver crashed ({e})")
            return np.asarray(u_ref), np.asarray(z_ref), False

        # Solved to tolerance, or to reduced accuracy (cvxpy's OPTIMAL and OPTIMAL_INACCURATE)
        if str(res.status) in ("Solved", "AlmostSolved"):
            self.result = res
            x = np.asarray(res.x)
            return x[self.o_u:self.o_mp].reshape(H, self.d_u), x[:self.n_z].reshape(H + 1, d_z), True
        print(f"SCP subproblem: solver returned status {res.status}")
        return np.asarray(u_ref), np.asarray(z_ref), False