"""Check a trained world model's goal head: does its goal probability match how often the goal is really reached?

The dataset has no goal flag, so a row counts as reaching the goal when its reward is above goal_threshold. Every row
is scored twice: from the encoding of its real observation, and from the latent the dynamics predict after rolling the
dataset's actions forward --steps rows within the same trajectory, which is what the planners see. For each it prints
the mean probability on goal rows and on the rest, the expected against the real number of goal events, precision and
recall at 0.5, and a calibration table (mean probability against the real goal rate in probability bins). The model
was trained on the whole dataset, so these are in-sample numbers.

usage: python src/scripts/check_goal_head.py configs/wmTraining/pg1_c_1_5_model.yaml [--steps 9]
"""
import argparse
import sys
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import orbax.checkpoint as ocp
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from safe_jepa.world_model.world_model import WorldModel
from train_world_model import resolve_dims, episode_ids

BINS = [0.0, 0.01, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

def load_trained(cfg_filename):
    # Rebuild the world model from its training config and restore the saved weights (as sweep_runner does)
    cfg = yaml.load(open(ROOT / cfg_filename), Loader=yaml.FullLoader)
    with h5py.File(ROOT / cfg["dataset_filename"], 'r') as f:
        data = {k: f[k][:] for k in f.keys()}
    cfg = resolve_dims(cfg, data["observations"].shape[-1], data["actions"].shape[-1])
    obs_dim = data["observations"].shape[-1]
    world_model = WorldModel(cfg, jnp.zeros(obs_dim), jnp.ones(obs_dim), rngs=nnx.Rngs(cfg["seed"]))
    _, target = nnx.split(world_model)
    ckpt_path = ROOT / "src/save_data" / cfg["data_dir"] / "Model/state"
    nnx.update(world_model, ocp.StandardCheckpointer().restore(ckpt_path, target))
    return world_model, cfg, data

def goal_probability(world_model, data, rows, steps):
    # Latent at each row: the encoding of its observation (steps = 0), or the dynamics' prediction after rolling the
    # dataset's actions forward from `steps` rows earlier; then the goal probability the planners see
    reward_fn = world_model.reward_fn
    p = []
    for chunk in np.array_split(rows, max(1, len(rows) // 65536)):
        z = world_model.encoder(jnp.asarray(data["observations"][chunk - steps]))
        for k in range(steps, 0, -1):
            z = world_model.dynamics(z, jnp.asarray(data["actions"][chunk - k]))
        _, logit = reward_fn.heads(z, jnp.asarray(data["actions"][chunk]))
        p.append(np.asarray(jax.nn.sigmoid(logit - reward_fn.log_goal_weight)))
    return np.concatenate(p)

def report(label, p, goal):
    predicted = p > 0.5
    precision = goal[predicted].mean() if predicted.any() else float("nan")
    print(f"{label}: {goal.sum()} goal rows of {len(goal)} | mean p on goal rows {p[goal].mean():.3f}, on the rest "
          f"{p[~goal].mean():.4f} | expected goal events {p.sum():.0f} vs real {goal.sum()} | "
          f"precision {precision:.3f}, recall {predicted[goal].mean():.3f} at p > 0.5")
    print("    p bin          rows    mean p   real goal rate")
    bins = np.digitize(p, BINS[1:-1])
    for b in range(len(BINS) - 1):
        sel = bins == b
        if sel.any():
            print(f"    {BINS[b]:.2f}-{BINS[b + 1]:.2f} {sel.sum():10d}    {p[sel].mean():.4f}   {goal[sel].mean():.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wm_cfg")
    parser.add_argument("--steps", type=int, default=9)
    args = parser.parse_args()

    world_model, cfg, data = load_trained(args.wm_cfg)
    assert world_model.reward_fn.goal_head, "this model has no goal head (its config has no goal_weight)"
    goal = data["rewards"] > cfg.get("goal_threshold", 0.5)
    episode = episode_ids(data)
    n = len(goal)

    print(f"{args.wm_cfg} ({cfg['data_dir']})")
    for steps in (0, args.steps):
        # Rows with `steps` earlier rows in the same trajectory
        rows = np.arange(steps, n)
        rows = rows[episode[rows - steps] == episode[rows]]
        p = goal_probability(world_model, data, rows, steps)
        report("from the encoded observation" if steps == 0 else f"after {steps} predicted steps", p, goal[rows])

if __name__ == "__main__":
    main()
