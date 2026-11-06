import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

FILTER_ITERS = 5 # re-linearisations of the barrier around the corrected action
FILTER_TOL = 1e-4 # barrier risk this far above the threshold counts as met (SCP's feas_tol)

@nnx.jit
def barrier_and_grad(Q_fn, z, u):
    # The barrier risk Q(z, u) = (1 - alpha) h(z) - h(f(z, u)) at the current latent, and its gradient in the action
    return jax.value_and_grad(lambda v: Q_fn(z, v, update_spectral_norm=False).squeeze())(u)

def project(u_nom, a, b, lo, hi):
    # argmin ||u - u_nom||^2 subject to a.u <= b and lo <= u <= hi. Its solution is u(lam) = clip(u_nom - lam a) for the
    # smallest lam >= 0 with a.u(lam) <= b; a.u(lam) never increases with lam, so lam is found by bisection. If no
    # action in the box meets a.u <= b, the box corner that makes a.u smallest (the largest decrease in Q) is returned
    u = np.clip(u_nom, lo, hi)
    if a @ u <= b:
        return u
    corner = np.where(a > 0, lo, np.where(a < 0, hi, u))
    if a @ corner > b:
        return corner
    lam_lo, lam_hi = 0.0, 1.0
    while a @ np.clip(u_nom - lam_hi * a, lo, hi) > b:
        lam_hi *= 2.0
    for _ in range(60):
        lam = 0.5 * (lam_lo + lam_hi)
        lam_lo, lam_hi = (lam, lam_hi) if a @ np.clip(u_nom - lam * a, lo, hi) > b else (lam_lo, lam)
    return np.clip(u_nom - lam_hi * a, lo, hi)

def safety_filter(Q_fn, z, u_nom, lo, hi, tau=0.0):
    # One-step barrier safety filter (as in Ames et al. 2019, in discrete time): the action closest to the planner's
    # u_nom whose next step meets Q(z, u) <= tau. Q is linearised around the current action and the projection repeated
    # up to FILTER_ITERS times; where no action in the box meets the linearised condition, the one that lowers Q most
    # is taken. Only the executed action changes; the planner's plan and its warm start are left as they were
    u_nom = np.asarray(u_nom, dtype=np.float64)
    u, q_before = np.clip(u_nom, lo, hi), None
    for _ in range(FILTER_ITERS):
        q, a = barrier_and_grad(Q_fn, z, jnp.asarray(u, dtype=jnp.float32))
        q, a = float(q), np.asarray(a, dtype=np.float64)
        q_before = q if q_before is None else q_before
        if q <= tau + FILTER_TOL or not np.any(a):
            break
        u = project(u_nom, a, tau - q + a @ u, lo, hi) # Q(u_i) + a.(u - u_i) <= tau
    else:
        q = float(barrier_and_grad(Q_fn, z, jnp.asarray(u, dtype=jnp.float32))[0])
    return u, {"filter_active": q_before > tau + FILTER_TOL, "filter_q_before": q_before, "filter_q_after": q,
               "filter_shift": float(np.linalg.norm(u - u_nom))}
