import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from analyse_episodes import load_world_model
from train_world_model import episode_ids, margin_labels, margin_constraints

N_WINDOWS = 2000 # windows of H + 1 consecutive dataset rows from one trajectory
H = 10 # rollout steps checked (the planning horizon)
N_ACTIONS = 64 # random actions tried from each latent

@nnx.jit
def _encode(world_model, obs):
    return world_model.target_encoder(obs)

@nnx.jit
def _step(world_model, z, u):
    return world_model.dynamics(z, u)

@nnx.jit
def _margin(world_model, z):
    return world_model.margin_fn(z)

def check(cfg_path):
    cfg = yaml.safe_load(open(ROOT / cfg_path))
    world_model, data = load_world_model(cfg)
    ep = episode_ids(data)
    rng = np.random.default_rng(0)
    starts = rng.choice(np.flatnonzero(ep[:-H] == ep[H:]), N_WINDOWS, replace=False)
    z_real = [np.asarray(_encode(world_model, jnp.asarray(data["observations"][starts + k]))) for k in range(H + 1)]

    # Rollout error against predicting no change (z_k = z_0), and with the windows' action sequences shuffled between
    # windows: a model that uses the action does clearly worse with the wrong actions
    def rollout(rows):
        z, zs = jnp.asarray(z_real[0]), []
        for k in range(H):
            z = _step(world_model, z, jnp.asarray(data["actions"][rows + k]))
            zs.append(np.asarray(z))
        return zs
    z_model, z_shuffled = rollout(starts), rollout(rng.permutation(starts))
    err_model = [np.mean((z_model[k] - z_real[k + 1]) ** 2) for k in range(H)]
    err_shuffled = [np.mean((z_shuffled[k] - z_real[k + 1]) ** 2) for k in range(H)]
    err_still = [np.mean((z_real[0] - z_real[k + 1]) ** 2) for k in range(H)]
    print(f"\n{cfg['data_dir']}: {N_WINDOWS} windows of {H} steps")
    print("  rollout MSE per latent dim, model vs predicting no change vs model with shuffled actions: " + ", ".join(
        f"step {k + 1} {err_model[k]:.4f} vs {err_still[k]:.4f} vs {err_shuffled[k]:.4f}" for k in (0, 1, 4, H - 1)))

    # What the barrier sees along a plan: the margin head on the rolled-out latents against the real margins (as
    # trained, clipped), on all rows and on rows near a constraint's edge
    constraints = margin_constraints(cfg)
    m_real = [margin_labels(data["observations"][starts + k + 1], constraints) for k in range(H)]
    m_pred = [np.asarray(_margin(world_model, jnp.asarray(z_model[k]))) for k in range(H)]
    for c, con in enumerate(constraints):
        line = []
        for k in (0, 1, 4, H - 1):
            d = m_pred[k][:, c] - m_real[k][:, c]
            near = m_real[k][:, c] < 0.3 * con["scale"]
            line.append(f"step {k + 1} {d.mean():+.3f}/{np.sqrt(np.mean(d ** 2)):.3f} | "
                        f"{d[near].mean():+.3f}/{np.sqrt(np.mean(d[near] ** 2)):.3f}")
        print(f"  margin {c} ({con['label']}) on the rollout, bias/RMS error on all rows | on rows with real margin "
              f"< 0.3 scale: " + ", ".join(line))

    # How much the predicted next latent depends on the action: its spread over random actions from the same latents
    z0 = jnp.asarray(z_real[0][:500])
    lo, hi = cfg["action_bounds"]
    u_rand = rng.uniform(lo, hi, (N_ACTIONS, len(z0), data["actions"].shape[-1])).astype(np.float32)
    outs = np.stack([np.asarray(_step(world_model, z0, jnp.asarray(u))) for u in u_rand])
    print(f"  spread of the next latent over {N_ACTIONS} random actions {np.sqrt(np.mean(np.var(outs, axis=0))):.4f} "
          f"per dim, against the real one-step change {np.sqrt(err_still[0]):.4f} and the one-step error "
          f"{np.sqrt(err_model[0]):.4f}")

    # Largest singular values of df/du and of df/dz - I (the residual's slope in z)
    f = lambda z, u: world_model.dynamics(z[None], u[None])[0]
    Jz, Ju = jax.vmap(jax.jacfwd(f, argnums=(0, 1)))(z0[:200], jnp.asarray(data["actions"][starts[:200]]))
    top = lambda J: np.linalg.svd(np.asarray(J), compute_uv=False)[:, 0]
    print(f"  median largest singular value: df/du {np.median(top(Ju)):.3f}, df/dz - I "
          f"{np.median(top(Jz - jnp.eye(Jz.shape[-1]))):.3f} "
          f"(g is bounded by {cfg['DynamicsParams']['lipschitz_bound'] - 1.0})")

if __name__ == "__main__":
    for cfg_path in ["configs/wmTraining/pg1_c_1_5_model.yaml"]:
        check(cfg_path)