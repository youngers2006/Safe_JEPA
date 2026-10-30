import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import orbax.checkpoint as ocp
import h5py
import yaml
from functools import partial
from tqdm import tqdm
from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from safe_jepa.world_model.world_model import WorldModel
from safe_jepa.nn.layers import SpectralNormLinear

VAL_BATCHES = 50

def resolve_dims(cfg, obs_dim, act_dim):
    d_z = cfg["d_latent"]
    cfg["EncoderParams"]["obs_type"] = cfg["obs_type"]
    cfg["EncoderParams"]["obs_dim"] = obs_dim
    cfg["EncoderParams"]["d_latent"] = d_z
    cfg["EncoderParams"]["obs_keep"] = [i for i in range(obs_dim) if i not in cfg["obs_drop"]]
    cfg["DynamicsParams"]["d_in"] = d_z + act_dim
    cfg["DynamicsParams"]["d_out"] = d_z
    cfg["RewardParams"]["d_in"] = d_z + act_dim
    cfg["RewardParams"]["d_out"] = 1
    cfg["ValueParams"]["d_in"] = d_z
    cfg["ValueParams"]["d_out"] = 1
    cfg["ValueQParams"]["d_in"] = d_z + act_dim
    cfg["ValueQParams"]["d_out"] = 1
    cfg["margin_constraints"] = margin_constraints(cfg)
    cfg["MarginParams"]["d_in"] = d_z
    cfg["MarginParams"]["d_out"] = len(cfg["margin_constraints"])
    for sub in ("DynamicsParams", "ValueParams", "ValueQParams", "RewardParams", "MarginParams"):
        cfg[sub]["hidden_features"] = tuple(cfg[sub]["hidden_features"])
    return cfg

def write_metrics(history: list[list[dict]], path: str | Path) -> None:
    """history[epoch][batch] -> dict of scalar metrics."""
    history = jax.device_get(history)

    rows, step = [], 0
    for e, epoch in enumerate(history):
        for b, m in enumerate(epoch):
            rows.append({"epoch": e, "batch": b, "step": step,
                         **{k: float(v) for k, v in m.items()}})
            step += 1

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def safety_margin(obs, sources):
    return np.min([3.0 * (1.0 - obs[:, lo:hi].max(axis=1)) - contact for lo, hi, contact in sources], axis=0)

def margin_constraints(cfg):
    # The margin head's constraints; configs from before margin_constraints have one, built from margin_sources
    return cfg.get("margin_constraints") or [{"label": "lidar", "sources": cfg["margin_sources"], "scale": 1.0}]

# How each kind of constraint's margin is computed from observations: label name -> fn(obs, constraint config)
MARGIN_LABELS = {
    "lidar": lambda obs, c: safety_margin(obs, c["sources"]),
}

def margin_labels(obs, constraints):
    # The margin head's targets, one column per constraint in the order of margin_constraints, each clipped at the
    # constraint's clip: margins further from its edge than this are trained as this
    return np.stack([
        np.minimum(MARGIN_LABELS[c["label"]](obs, c), c.get("clip", np.inf)) for c in constraints
    ], axis=-1)

def episode_ids(data_dict):
    ends = data_dict["terminals"].astype(bool) | data_dict["timeouts"].astype(bool)
    joins = np.all(
        np.isclose(data_dict["next_observations"][:-1], data_dict["observations"][1:]), 
        axis=-1
    )
    return np.concatenate([[0], np.cumsum(~(~ends[:-1] & joins))])

def val_rows(episode, cfg):
    # Rows of the trajectories held out for validation: a val_frac share of the trajectories, chosen by the seed, so no
    # validation window shares a trajectory with a training one
    n_episodes = int(episode[-1]) + 1
    n_val = int(round(cfg["val_frac"] * n_episodes))
    held_out = np.random.default_rng(cfg["seed"]).choice(n_episodes, n_val, replace=False)
    return np.isin(episode, held_out)

@partial(jax.jit, static_argnames=("batch_size", "horizon"))
def _sample(data_dict, key, batch_size, horizon):
    n = data_dict["observations"].shape[0]
    starts = jax.random.randint(key, (batch_size,), 0, n)
    rows = starts[:, None] + jnp.arange(horizon)
    idx = jnp.minimum(rows, n - 1)
    batch = {k: jnp.asarray(v[idx]) for k, v in data_dict.items() if k != "episode"}

    same_episode = data_dict["episode"][idx] == data_dict["episode"][starts][:, None]
    batch["mask"] = ((rows < n) & same_episode).astype(jnp.float32)
    return batch, starts

def get_batches(data_dict, batch_size, N, key, epoch, horizon):
    for i in range(N):
        yield _sample(data_dict, jax.random.fold_in(key, i + epoch * N), batch_size, horizon)

def validate(world_model, val_dict, key, batch_size, horizon):
    # Mean of each training loss over VAL_BATCHES held-out batches, without changing the model
    metrics = [world_model.train_step(b["observations"], b["next_observations"], b["actions"], b["rewards"],
                                      b["margins"], b["terminals"], b["mask"], update=False)
               for b, _ in get_batches(val_dict, batch_size, VAL_BATCHES, key, 0, horizon)]
    return {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}

def main(cfg_filename):
    # Load training config
    with open(ROOT / cfg_filename, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Initialise datadict
    data_dict = {}

    # Load dataset (assume dataset has been processed)
    with h5py.File(ROOT / cfg["dataset_filename"], 'r') as f:
        for k in list(f.keys()):
            data_dict[k] = f[k][:]

    # Augment datadict
    data_dict["episode"] = episode_ids(data_dict)
    print(f"{data_dict['episode'][-1] + 1} trajectories, {data_dict['observations'].shape[0]} transitions")
    data_dict["margins"] = margin_labels(data_dict["next_observations"], margin_constraints(cfg))
    for k, c in enumerate(margin_constraints(cfg)):
        m_k = data_dict["margins"][:, k]
        print(f"margin {k} ({c['label']}) <= 0 on {np.mean(m_k[data_dict['costs'] > 0] <= 0):.1%} of cost rows "
              f"and {np.mean(m_k[data_dict['costs'] == 0] <= 0):.1%} of cost-free rows")
    goal_rows = data_dict["rewards"] > cfg["goal_threshold"]
    print(f"{goal_rows.mean():.2%} goal rows; largest |reward| on the others "
          f"{np.abs(data_dict['rewards'][~goal_rows]).max():.3f}, smallest goal-row reward "
          f"{data_dict['rewards'][goal_rows].min():.3f}")

    # Hold out whole trajectories for validation
    val = val_rows(data_dict["episode"], cfg)
    val_dict = {k: v[val] for k, v in data_dict.items()}
    data_dict = {k: v[~val] for k, v in data_dict.items()}
    print(f"validation: {len(np.unique(val_dict['episode']))} trajectories held out ({val.mean():.1%} of transitions)")

    # Add additional data
    cfg = resolve_dims(
        cfg,
        data_dict["observations"].shape[-1],
        data_dict["actions"].shape[-1]
    )

    # Create rng key
    data_key = jax.random.key(cfg["seed"])
    rngs = nnx.Rngs(cfg["seed"])

    # Calculate observation distribution
    obs_mean = jnp.mean(data_dict["observations"], axis=0)
    obs_std = jnp.std(data_dict["observations"], axis=0)

    # Setup world model
    world_model = WorldModel(
        cfg,
        obs_mean,
        obs_std,
        rngs=rngs
    )

    # Transfer data
    data_dict = {k: jnp.asarray(v) for k, v in data_dict.items()}
    val_dict = {k: jnp.asarray(v) for k, v in val_dict.items()}
    val_key = jax.random.key(cfg["seed"] + 1)

    # Run training loop
    print("Beginning Training Loop ... ")
    horizon = cfg["train_horizon"]
    metrics_log = []
    val_log = []
    for epoch in tqdm(range(cfg["epochs"]), desc=f"Epochs", leave=True):
        metrics_log_epoch = []
        for i, (batch, _) in enumerate(tqdm(get_batches(data_dict, cfg["batch_size"], 
                cfg["num_batches"], data_key, epoch, horizon), total=cfg["num_batches"], desc=f"Batches", leave=False)):
            obs = batch["observations"]
            next_obs = batch["next_observations"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            terminals = batch["terminals"]
            mask = batch["mask"]
            margins = batch["margins"]
            metrics = world_model.train_step(
                obs, next_obs, actions, rewards, margins, terminals, mask)
            metrics_log_epoch.append(jax.device_get(metrics))
        metrics_log.append(metrics_log_epoch)
        val_log.append([validate(world_model, val_dict, val_key, cfg["batch_size"], horizon)])
        tqdm.write(f"epoch {epoch} validation: " + ", ".join(f"{k} {val_log[-1][0][k]:.3g}" for k in (
            "loss_total", "loss_dyn", f"loss_dyn_t{horizon - 1}", "loss_margin", "loss_r", "loss_v")))
    print("Training Loop Completed.")

    print("Saving Metrics")
    # Save training metrics
    save_path_metrics = ROOT / "src/save_data" / cfg["data_dir"] / "Metrics"
    save_path_metrics.mkdir(parents=True, exist_ok=True)
    write_metrics(metrics_log, save_path_metrics / "metrics.csv")
    write_metrics(val_log, save_path_metrics / "val_metrics.csv")
    print("Metrics Saved")

    print("Saving Model")
    # Save the trained model
    for _, node in nnx.iter_graph(world_model):
        if isinstance(node, SpectralNormLinear):
            node.set_exact_uv()
    _, state = nnx.split(world_model)
    checkpointer = ocp.StandardCheckpointer()
    save_path_model = ROOT / "src/save_data" / cfg["data_dir"] / "Model"
    save_path_model.mkdir(parents=True, exist_ok=True)
    checkpointer.save(save_path_model / 'state', state)
    checkpointer.wait_until_finished()
    print("Model Saved")

    print(f"Training Complete")
    return 0

if __name__ == "__main__":
    main("configs/wmTraining/pg1_c_1_5_model.yaml")
    main("configs/wmTraining/pg1_c_1_2_model.yaml")
    main("configs/wmTraining/pg1_c_inf_model.yaml")