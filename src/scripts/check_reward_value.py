import sys
from pathlib import Path

import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from analyse_episodes import load_world_model
from train_world_model import episode_ids

N_ROWS = 200_000 # dataset rows checked
CHUNK = 8192

@nnx.jit
def _heads(world_model, obs, act):
    z = world_model.encoder(obs)
    return world_model.reward_fn(z, act).squeeze(-1), world_model.value_fn(z).squeeze(-1)

def return_to_go(rewards, terminals, episode, gamma):
    # Discounted return actually earned from each row to the end of its trajectory (nothing after a terminal)
    g = np.zeros(len(rewards))
    for i in range(len(rewards) - 1, -1, -1):
        same = i + 1 < len(rewards) and episode[i + 1] == episode[i] and not terminals[i]
        g[i] = rewards[i] + (gamma * g[i + 1] if same else 0.0)
    return g

def check(cfg_path):
    cfg = yaml.safe_load(open(ROOT / cfg_path))
    world_model, data = load_world_model(cfg)
    n = len(data["rewards"])
    idx = np.sort(np.random.default_rng(0).choice(n, min(N_ROWS, n), replace=False))
    r_pred, v_pred = [], []
    for i in range(0, len(idx), CHUNK):
        rows = idx[i:i + CHUNK]
        rp, vp = _heads(world_model, jnp.asarray(data["observations"][rows]), jnp.asarray(data["actions"][rows]))
        r_pred.append(np.asarray(rp)); v_pred.append(np.asarray(vp))
    r_pred, v_pred = np.concatenate(r_pred), np.concatenate(v_pred)
    r = data["rewards"][idx]
    g = return_to_go(data["rewards"], data["terminals"].astype(bool), episode_ids(data), cfg["discount"])[idx]

    goal = r > cfg["goal_threshold"] # the goal bonus lifts these rows; the rest are the step's progress
    dense = ~goal
    r_target = r - cfg["goal_reward"] * goal if cfg["dense_reward"] else r # what the reward head was trained on
    print(f"\n{cfg['data_dir']}: {len(idx)} dataset rows, {goal.mean():.2%} with the goal bonus")
    print(f"  reward: MSE {np.mean((r_pred - r_target) ** 2):.2e} against its target's variance {r_target.var():.2e} "
          f"(MSE equal to the variance = predicting a constant)")
    print(f"  reward, dense rows: correlation {np.corrcoef(r_pred[dense], r[dense])[0, 1]:+.3f}, "
          f"RMS error {np.sqrt(np.mean((r_pred[dense] - r[dense]) ** 2)):.4f} against std {r[dense].std():.4f}, "
          f"sign right on {np.mean(np.sign(r_pred[dense]) == np.sign(r[dense])):.1%}")
    print(f"  reward, goal rows: predicted mean {r_pred[goal].mean():+.3f} (real 1 + progress), "
          f"other rows' predicted mean {r_pred[dense].mean():+.4f}")
    print(f"  value against the discounted return earned: correlation {np.corrcoef(v_pred, g)[0, 1]:+.3f}, "
          f"bias {np.mean(v_pred - g):+.3f}, RMS error {np.sqrt(np.mean((v_pred - g) ** 2)):.3f}, "
          f"return mean {g.mean():.3f} std {g.std():.3f}")

if __name__ == "__main__":
    for cfg_path in ["configs/wmTraining/pg1_c_1_5_model.yaml"]:
        check(cfg_path)