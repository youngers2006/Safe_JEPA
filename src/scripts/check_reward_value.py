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
    dense = world_model.reward_fn(z, act).squeeze(-1)
    goal_logit = world_model.goal_fn(z, act).squeeze(-1) if world_model.goal_head else jnp.zeros_like(dense)
    return dense, goal_logit, world_model.value_fn(z).squeeze(-1)

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
    dense_pred, goal_logit, v_pred = [], [], []
    for i in range(0, len(idx), CHUNK):
        rows = idx[i:i + CHUNK]
        dp, gl, vp = _heads(world_model, jnp.asarray(data["observations"][rows]), jnp.asarray(data["actions"][rows]))
        dense_pred.append(np.asarray(dp)); goal_logit.append(np.asarray(gl)); v_pred.append(np.asarray(vp))
    dense_pred, goal_logit, v_pred = np.concatenate(dense_pred), np.concatenate(goal_logit), np.concatenate(v_pred)

    # The reward as the planners see it: with the goal head, the dense reward plus the goal bonus times its probability
    p_goal = None
    r_pred = dense_pred
    if world_model.goal_head:
        p_goal = 1.0 / (1.0 + np.exp(-(goal_logit - np.log(world_model.goal_weight))))
        r_pred = dense_pred + world_model.goal_reward * p_goal
    r = data["rewards"][idx]
    g = return_to_go(data["rewards"], data["terminals"].astype(bool), episode_ids(data), cfg["discount"])[idx]

    goal = r > 0.5 # the goal bonus is +1; the dense reward is the step's progress towards the goal
    dense = ~goal
    print(f"\n{cfg['data_dir']}: {len(idx)} dataset rows, {goal.mean():.2%} with the goal bonus")
    print(f"  reward: MSE {np.mean((r_pred - r) ** 2):.2e} against the reward's variance {r.var():.2e} "
          f"(MSE equal to the variance = predicting a constant)")
    print(f"  reward, dense rows: correlation {np.corrcoef(r_pred[dense], r[dense])[0, 1]:+.3f}, "
          f"RMS error {np.sqrt(np.mean((r_pred[dense] - r[dense]) ** 2)):.4f} against std {r[dense].std():.4f}, "
          f"sign right on {np.mean(np.sign(r_pred[dense]) == np.sign(r[dense])):.1%}")
    print(f"  reward, goal rows: predicted mean {r_pred[goal].mean():+.3f} (real 1 + progress), "
          f"other rows' predicted mean {r_pred[dense].mean():+.4f}")
    if p_goal is not None:
        print(f"  goal head: mean probability {p_goal[goal].mean():.3f} on goal rows and {p_goal[dense].mean():.4f} on the "
              f"rest; probability over 0.5 on {np.mean(p_goal[goal] > 0.5):.1%} of goal rows and {np.mean(p_goal[dense] > 0.5):.2%} "
              f"of the rest")
    print(f"  value against the discounted return earned: correlation {np.corrcoef(v_pred, g)[0, 1]:+.3f}, "
          f"bias {np.mean(v_pred - g):+.3f}, RMS error {np.sqrt(np.mean((v_pred - g) ** 2)):.3f}, "
          f"return mean {g.mean():.3f} std {g.std():.3f}")

if __name__ == "__main__":
    for cfg_path in ["configs/wmTraining/pg1_c_1_5_model.yaml"]:
        check(cfg_path)