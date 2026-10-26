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

def resolve_dims(cfg, obs_dim, act_dim):
    d_z = cfg["d_latent"]
    cfg["EncoderParams"]["obs_type"] = cfg["obs_type"]
    cfg["EncoderParams"]["obs_dim"] = obs_dim
    cfg["EncoderParams"]["d_latent"] = d_z
    cfg["DynamicsParams"]["d_in"] = d_z + act_dim
    cfg["DynamicsParams"]["d_out"] = d_z
    cfg["RewardParams"]["d_in"] = d_z + act_dim
    cfg["RewardParams"]["d_out"] = 1
    cfg["ValueParams"]["d_in"] = d_z
    cfg["ValueParams"]["d_out"] = 1
    cfg["ValueQParams"]["d_in"] = d_z + act_dim
    cfg["ValueQParams"]["d_out"] = 1
    cfg["SafetyCriticParams"]["d_in"] = d_z + act_dim
    cfg["SafetyCriticParams"]["d_out"] = 1
    for sub in ("DynamicsParams", "ValueParams", "ValueQParams", "RewardParams", "SafetyCriticParams"):
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

def episode_ids(data_dict):
    ends = data_dict["terminals"].astype(bool) | data_dict["timeouts"].astype(bool)
    joins = np.all(
        np.isclose(data_dict["next_observations"][:-1], data_dict["observations"][1:]), 
        axis=-1
    )
    return np.concatenate([[0], np.cumsum(~(~ends[:-1] & joins))])

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

    data_dict["episode"] = episode_ids(data_dict)
    print(f"{data_dict['episode'][-1] + 1} trajectories, {data_dict['observations'].shape[0]} transitions")

    # Add additional data
    cfg = resolve_dims(
        cfg,
        data_dict["observations"].shape[-1],
        data_dict["actions"].shape[-1]
    )

    # Create rng key
    mask_key, data_key, train_key = jax.random.split(jax.random.key(cfg["seed"]), 3)
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

    # Run training loop
    print("Beginning Training Loop ... ")
    horizon = cfg["train_horizon"]
    metrics_log = []
    for epoch in tqdm(range(cfg["epochs"]), desc=f"Epochs", leave=True):
        metrics_log_epoch = []
        for i, (batch, _) in enumerate(tqdm(get_batches(data_dict, cfg["batch_size"], 
                cfg["num_batches"], data_key, epoch, horizon), total=cfg["num_batches"], desc=f"Batches", leave=False)):
            obs = batch["observations"]
            next_obs = batch["next_observations"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            safety_costs = batch["costs"]
            terminals = batch["terminals"]
            mask = batch["mask"]
            metrics = world_model.train_step(
                obs, next_obs, actions, rewards, safety_costs, terminals, mask,
                jax.random.fold_in(train_key, epoch * cfg["num_batches"] + i)
            )
            metrics_log_epoch.append(jax.device_get(metrics))
        metrics_log.append(metrics_log_epoch)
    print("Training Loop Completed.")

    print("Saving Metrics")
    # Save training metrics
    save_path_metrics = ROOT / "src/save_data" / cfg["data_dir"] / "Metrics"
    save_path_metrics.mkdir(parents=True, exist_ok=True)
    write_metrics(metrics_log, save_path_metrics / "metrics.csv")
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
    main("configs/wmTraining/pg1_c_inf_model.yaml")