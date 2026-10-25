import numpy as np
import scipy.sparse as sp
import osqp

class SCPSolver():
    def __init__(self, horizon: int, d_z: int, d_u: int, discount: float, u_min: float = -1.0, u_max: float = 1.0):
        # Constants
        self.horizon = horizon
        self.d_z = d_z
        self.d_u = d_u
        self.discount = discount
        self.u_min, self.u_max = u_min, u_max

        # The convex subproblem is the QP
        #   min  sum_h -gamma^h (Jr_z z_h + Jr_u u_h) - gamma^H Jv_z z_H + rho_u ||u||^2
        #        + w_prox (||z_1:H - z_ref||^2 + ||u - u_ref||^2) + w_slack sum(mu_p + mu_n + nu)
        #   s.t. z_0 = z_c, z_h+1 = A_h z_h + B_h u_h + r_h + mu_p,h - mu_n,h, C_h z_h + D_h u_h - nu_h <= tau - r'_h,
        #        u_min <= u <= u_max, mu_p, mu_n, nu >= 0
        # posed for OSQP as min 1/2 x'Px + q'x s.t. l <= Ax <= u over x = (z_0..z_H, u_0..u_H-1, mu_p, mu_n, nu), built once
        # with fixed sparsity and updated in place each iteration
        H = horizon
        self.n_z, self.n_u, self.n_mu = (H + 1) * d_z, H * d_u, H * d_z
        self.o_u, self.o_mp = self.n_z, self.n_z + self.n_u
        self.o_mn, self.o_nu = self.o_mp + self.n_mu, self.o_mp + 2 * self.n_mu
        self.n = self.o_nu + H
        z_col = lambda h, i: h * d_z + i
        u_col = lambda h, j: self.o_u + h * d_u + j

        # Constraint rows: z_0 = z_c, dynamics z_h+1 - A z_h - B u_h - mu_p + mu_n = r, safety C z_h + D u_h - nu <= tau - r',
        # action bounds, slack >= 0. Entries that change each iteration come first, in the order of A, B, C, D flattened
        r_dyn, r_safe = d_z, d_z + H * d_z
        r_act = r_safe + H
        r_slack = r_act + self.n_u
        self.m = r_slack + 2 * self.n_mu + H
        rows, cols = [], []
        for h in range(H):                                       # -A_h on z_h
            for i in range(d_z):
                rows += [r_dyn + h * d_z + i] * d_z; cols += [z_col(h, j) for j in range(d_z)]
        for h in range(H):                                       # -B_h on u_h
            for i in range(d_z):
                rows += [r_dyn + h * d_z + i] * d_u; cols += [u_col(h, j) for j in range(d_u)]
        for h in range(H):                                       # C_h on z_h
            rows += [r_safe + h] * d_z; cols += [z_col(h, j) for j in range(d_z)]
        for h in range(H):                                       # D_h on u_h
            rows += [r_safe + h] * d_u; cols += [u_col(h, j) for j in range(d_u)]
        self.n_varying = len(rows)
        const = []
        for i in range(d_z):                                     # z_0 = z_c
            rows.append(i); cols.append(z_col(0, i)); const.append(1.0)
        for h in range(H):
            for i in range(d_z):                                 # + z_h+1, - mu_p, + mu_n
                row = r_dyn + h * d_z + i
                rows += [row] * 3; cols += [z_col(h + 1, i), self.o_mp + h * d_z + i, self.o_mn + h * d_z + i]
                const += [1.0, -1.0, 1.0]
            rows.append(r_safe + h); cols.append(self.o_nu + h); const.append(-1.0)   # - nu
        for k in range(self.n_u):                                # action bounds
            rows.append(r_act + k); cols.append(self.o_u + k); const.append(1.0)
        for k in range(2 * self.n_mu + H):                       # slack >= 0
            rows.append(r_slack + k); cols.append(self.o_mp + k); const.append(1.0)
        self.const_vals = np.array(const)

        # CSC structure, and where each entry above lands in its data array
        order = sp.csc_matrix((np.arange(len(rows), dtype=float) + 1.0, (rows, cols)), shape=(self.m, self.n))
        self.A_perm = order.data.astype(int) - 1
        self.A_indices, self.A_indptr = order.indices, order.indptr

        # Fixed bounds: action bounds and slack >= 0
        self.l, self.u_bnd = np.zeros(self.m), np.zeros(self.m)
        self.l[r_safe:r_act] = -np.inf
        self.l[r_act:r_slack], self.u_bnd[r_act:r_slack] = u_min, u_max
        self.l[r_slack:], self.u_bnd[r_slack:] = 0.0, np.inf
        self.rows = (r_dyn, r_safe, r_act)

        # Quadratic terms: trust region on z_1..z_H, trust region plus action cost on u (diagonal, upper triangle)
        self.P_idx = np.arange(d_z, self.o_mp)
        self.solver, self.result = None, None

    def slack_and_multipliers(self) -> tuple[float, float, float, float]:
        # Largest slack used and largest constraint multiplier at the last solution. The L1 penalty caps each multiplier
        # at w_slack and a multiplier at the cap means its slack is in use, so w_slack must sit above the largest one.
        # Unpolished multipliers are only as accurate as OSQP's tolerances, so they are reported as NaN
        x, y = self.result.x, self.result.y
        r_dyn, r_safe, r_act = self.rows
        polished = self.result.info.status_polish == 1
        return (float(np.max(x[self.o_mp:self.o_mn] + x[self.o_mn:self.o_nu])), float(np.max(x[self.o_nu:])),
                float(np.max(np.abs(y[r_dyn:r_safe]))) if polished else np.nan,
                float(np.max(y[r_safe:r_act])) if polished else np.nan)

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
        H, d_z, d_u = self.horizon, self.d_z, self.d_u
        w_prox, rho_u, w_slack = float(hyperparams['w_prox']), float(hyperparams['rho_u']), float(hyperparams['w_slack'])
        r_dyn, r_safe, r_act = self.rows

        # Constraint matrix values, in the CSC order
        vals = np.concatenate([-np.asarray(A).ravel(), -np.asarray(B).ravel(), np.asarray(C).ravel(), np.asarray(D).ravel(),
                               self.const_vals])
        A_data = vals[self.A_perm]

        # Bounds: z_0 = z_c, dynamics equal to r, safety below tau - r'
        l, u = self.l.copy(), self.u_bnd.copy()
        l[:d_z] = u[:d_z] = z_c
        l[r_dyn:r_safe] = u[r_dyn:r_safe] = np.asarray(r).ravel()
        u[r_safe:r_act] = float(hyperparams['tau']) - np.asarray(r_prime)

        # Linear terms: rewards and terminal value, trust region centre, slack weights
        disc = self.discount ** np.arange(H)
        q = np.zeros(self.n)
        q[:H * d_z] -= (disc[:, None] * Jr_z).ravel()
        q[H * d_z:self.n_z] -= (self.discount ** H) * np.asarray(Jv_z)
        q[d_z:self.n_z] -= 2.0 * w_prox * np.asarray(z_ref)[1:].ravel()
        q[self.o_u:self.o_mp] = -(disc[:, None] * Jr_u).ravel() - 2.0 * w_prox * np.asarray(u_ref).ravel()
        q[self.o_mp:] = w_slack

        # Quadratic terms
        P_data = np.concatenate([np.full(self.n_z - d_z, 2.0 * w_prox), np.full(self.n_u, 2.0 * (w_prox + rho_u))])

        # Warm started from the last solution. Polishing recovers the exact solution on the active set; OSQP's tolerances
        # are relative to the cost terms, which grow with w_prox and w_slack, so 1e-6 keeps it reliable at large weights
        try:
            if self.solver is None:
                P = sp.csc_matrix((P_data, (self.P_idx, self.P_idx)), shape=(self.n, self.n))
                A_mat = sp.csc_matrix((A_data, self.A_indices, self.A_indptr), shape=(self.m, self.n))
                self.solver = osqp.OSQP()
                self.solver.setup(P, q, A_mat, l, u, verbose=False, polishing=True, warm_starting=True,
                                  eps_abs=1e-6, eps_rel=1e-6, max_iter=20000)
            else:
                self.solver.update(q=q, l=l, u=u, Px=P_data, Ax=A_data)
            res = self.solver.solve()
        except Exception as e:
            print(f"SCP subproblem: solver crashed ({e})")
            return np.asarray(u_ref), np.asarray(z_ref), False

        if res.info.status in ("solved", "solved inaccurate") and res.x is not None:
            self.result = res
            x = res.x
            return x[self.o_u:self.o_mp].reshape(H, d_u), x[:self.n_z].reshape(H + 1, d_z), True
        print(f"SCP subproblem: solver returned status {res.info.status}")
        return np.asarray(u_ref), np.asarray(z_ref), False
