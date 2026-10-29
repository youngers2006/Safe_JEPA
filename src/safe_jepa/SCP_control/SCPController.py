import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cp
from jax.experimental import enable_x64

from .SCPSolver import SCPSolver
from .get_jacobians import get_jacobians
from .nonlinear_cost_fn import nonlinear_cost_fn, rollout_fn
from ..Sampling_control.evaluate_sequences import plan_summary

class SCPController():
    def __init__(self, cfg: dict):
        # Problem dimensions
        self.horizon = cfg["horizon"]
        self.d_z = cfg["d_z"]
        self.d_u = cfg["d_u"]
        self.u_bounds = cfg["action_bounds"]

        # Objective weights, stored as a dict
        self.hyperparams = {
            "discount": cfg["discount"],
            "tau": cfg["safety_threshold"],
            "rho_u": cfg["rho_u"],
            "w_dyn": cfg["w_dyn"],
            "w_safe": cfg["w_safe"],
            "w_prox": cfg["w_prox"]
        }
        self.w_prox_init = cfg["w_prox"]
        self.w_prox_min = cfg["w_prox_min"]

        # w_prox carried between steps
        self.w_prox_warm = cfg["w_prox"]
        self.w_prox_max_start = 10 * cfg["w_prox"]

        # SCP loop settings
        self.scp_iters = cfg["scp_iters"]
        self.tol = cfg["tol"]
        self.feas_tol = cfg["feas_tol"]

        # TR dynamic resizing trigger bounds
        self.rho_lower = cfg["rho_lower"]
        self.rho_upper = cfg["rho_upper"]

        # Jitted function objects
        self.jac_extractor_fn = get_jacobians
        self.nonlinear_cost_fn = nonlinear_cost_fn
        self.rollout_fn = rollout_fn

        # Solver parameters
        self.SCP_solver = SCPSolver(
            self.horizon,
            self.d_z, 
            self.d_u, 
            self.hyperparams['discount'],
            self.hyperparams['rho_u'],
            self.u_bounds[0], 
            self.u_bounds[1]
        )

        # Flag to trigger initial warm start
        self.is_first_step = True
        
        # Memory buffer for trajectories to use in warm starting
        self.u_prev = np.zeros((self.horizon, self.d_u))
        self.z_prev = np.zeros((self.horizon + 1, self.d_z))

    def linear_cost_fn(self, system_matrices: tuple[np.ndarray, ...], z: np.ndarray, u: np.ndarray) -> np.ndarray:
        # Extract system matrices
        Jr_z, Jr_u, Jv_z, A, B, r, C, D, r_prime = system_matrices

        # Hyperparams
        gamma = self.hyperparams['discount']
        rho_u = self.hyperparams['rho_u']
        w_dyn = self.hyperparams['w_dyn']
        w_safe = self.hyperparams['w_safe']
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
        dyn_penalty = np.sum(np.abs(dyn_error))

        # nu usage penalty
        Cz = np.sum(C * z[:-1], axis=1)
        Du = np.sum(D * u, axis=1)
        safe_pred = Cz + Du + r_prime
        safe_penalty = np.sum(
            np.maximum(0.0, safe_pred - tau)
        )

        # Action magnitude penalty
        control_penalty = rho_u * np.sum(u ** 2)

        # Terminal value reward
        terminal_value = (gamma ** self.horizon) * np.sum(Jv_z * z[-1]) 

        # Compute total trajectory cost for the linearised system
        J_lin = (-discounted_reward 
                 + control_penalty 
                 + w_dyn * dyn_penalty 
                 + w_safe * safe_penalty 
                 - terminal_value)
        return float(J_lin)

    def rollout(self, z_c: jax.Array, u: np.ndarray, wm_networks: tuple[jax.Array, ...]):
        feasible, predictions = self.rollout_fn(
            z_c, 
            jnp.asarray(u),
            *wm_networks,
            self.hyperparams
        )
        z = np.concatenate([np.asarray(z_c)[np.newaxis], np.asarray(predictions["latents"])])
        return z, feasible, predictions

    def merit(self, z: np.ndarray, u: np.ndarray, wm_networks: tuple):
        J, max_defect, max_risk = self.nonlinear_cost_fn(
            jnp.asarray(z),
            jnp.asarray(u),
            *wm_networks,
            self.hyperparams
        )
        return float(J), float(max_defect), float(max_risk)

    def is_feasible(self, max_defect: float, max_risk: float) -> bool:
        return max_defect <= self.feas_tol and max_risk <= self.hyperparams['tau'] + self.feas_tol

    def _step(self, z_c: jax.Array, wm_networks: tuple[jax.Array, ...]) -> jax.Array:
        # Extract networks
        r_fn, v_fn, f_fn, Q_fn = wm_networks

        # Obtain w_prox from previous step
        self.hyperparams['w_prox'] = float(np.clip(self.w_prox_warm, self.w_prox_min, self.w_prox_max_start))

        if self.is_first_step:
            # Initial u is zeros
            u_ref = self.u_prev.copy()
            self.is_first_step = False
        else:
            # Shift actions by 1 and set last action to end of the prev action sequence
            u_ref = np.roll(self.u_prev, shift=-1, axis=0)
            u_ref[-1, :] = self.u_prev[-1, :]

        # Initialise trajectory by rolling u_ref out from the observation so z_ref obeys the dynamics
        z_ref, _, _ = self.rollout(z_c, u_ref, wm_networks)
        J_ref, defect_ref, risk_ref = self.merit(z_ref, u_ref, wm_networks)

        z_c_np = np.asarray(z_c)
        stop_reason = "max_iters"
        prox_grad = np.inf
        n_rejected = 0
        n_solver_failures = 0
        matrices = None
        iteration = -1
        for iteration in range(self.scp_iters):
            # Linearise only if the referene has changed
            if matrices is None:
                # Extract matrices from world model
                jax_matrices = self.jac_extractor_fn(
                    z_ref=jnp.array(z_ref), 
                    u_ref=jnp.array(u_ref),
                    r_fn=r_fn,
                    v_fn=v_fn,
                    f_fn=f_fn,
                    Q_fn=Q_fn
                )
                matrices = tuple(np.asarray(m) for m in jax_matrices)

            # Run solver to setup and solve SOCP
            u_opt, z_opt, ok = self.SCP_solver.solve_problem(
                matrices,
                z_c_np,
                z_ref,
                u_ref,
                self.hyperparams
            )

            # Failed solver step, reject and try again with a tighter trust region
            if not ok:
                n_solver_failures += 1
                self.hyperparams['w_prox'] = 2.0 * self.hyperparams['w_prox']
                continue

            # Second order Maratos correction
            z_sub = z_opt
            z_opt, _, _ = self.rollout(z_c, u_opt, wm_networks)

            # Convergence criteria
            # Prox-gradient norm |G_t|| = ||step|| / t with t = 1 / (2 w_prox)
            step_sq = np.sum((u_opt - u_ref) ** 2) + np.sum((z_opt[1:] - z_ref[1:]) ** 2)
            prox_grad = 2.0 * max(self.hyperparams['w_prox'], self.w_prox_init) * np.sqrt(step_sq)
            if prox_grad <= self.tol:
                stop_reason = "converged" if self.is_feasible(defect_ref, risk_ref) else "converged_infeasible"
                break

            # Compute reference trajectory costs with the model and linearised system
            linear_cost_old = self.linear_cost_fn(
                matrices, z_ref, u_ref
            )
            # Compute optimal trajectory costs with model and linearised system
            linear_cost_new = self.linear_cost_fn(
                matrices, z_opt, u_opt
            )
            J_new, defect_new, risk_new = self.merit(
                z_opt, u_opt, wm_networks
            )

            # Compute the ratio of cost improvement between true and linearised system
            dJ_actual = J_ref - J_new
            dJ_predicted = linear_cost_old - linear_cost_new
            rho = dJ_actual / dJ_predicted if (dJ_predicted > 0) else -np.inf

            # Trust region resizing
            if rho >= self.rho_lower:
                u_ref, z_ref = u_opt, z_opt
                J_ref, defect_ref, risk_ref = J_new, defect_new, risk_new
                matrices = None

                # If 
                if rho > self.rho_upper:
                    self.hyperparams['w_prox'] = max(
                        self.w_prox_min, 0.5 * self.hyperparams['w_prox']
                    )
            else:
                # Reject: keep the reference and shrink the step
                n_rejected += 1
                self.hyperparams['w_prox'] = 2.0 * self.hyperparams['w_prox']

        _, feasible_rollout, predictions = self.rollout(
            z_c, u_ref, wm_networks
        )
        certified = stop_reason == "converged" and bool(feasible_rollout)

        info = {
            "iterations": iteration + 1,
            "stop_reason": stop_reason,
            "certified": certified,
            "prox_grad": prox_grad,
            "max_defect": defect_ref,
            "w_prox_final": self.hyperparams['w_prox'],
            "n_rejected": n_rejected,
            "n_solver_failures": n_solver_failures,
            **plan_summary(jnp.asarray(u_ref), feasible_rollout, predictions),
        }
 
        # Update previous solution buffer and return 1st action to take
        self.u_prev = u_ref
        self.z_prev = z_ref
        self.w_prox_warm = self.hyperparams["w_prox"]
        return jnp.asarray(u_ref[0, :]), info

    def step(self, z_c: jax.Array, wm_networks: tuple[jax.Array, ...]) -> jax.Array:
        with enable_x64():
            u, info = self._step(jnp.asarray(z_c, dtype=jnp.float64), wm_networks)
        return jnp.asarray(u, dtype=jnp.float32), info