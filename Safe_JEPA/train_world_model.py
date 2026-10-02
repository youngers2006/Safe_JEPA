import jax
import jax.numpy as jnp
import flax.nnx as nnx
import orbax.checkpoint as ocp
import numpy as np
import h5py
import yaml
from functools import partial
from tqdm import tqdm
from pathlib import Path
import csv

from safe_wm.World_Model.WorldModel import WorldModel

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
    cfg["SafetyCriticParams"]["d_in"] = d_z + act_dim
    cfg["SafetyCriticParams"]["d_out"] = 1
    for sub in ("DynamicsParams", "ValueParams", "RewardParams", "SafetyCriticParams"):
        cfg[sub]["hidden_features"] = tuple(cfg[sub]["hidden_features"])
    return cfg

def write_metrics(history: list[list[dict]], path: str | Path) -> None:
    """history[epoch][batch] -> dict of scalar metrics."""
    history = jax.device_get(history)          # one sync for the whole tree

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

@partial(jax.jit, static_argnames=("batch_size",))
def _sample(data_dict, key, batch_size):
    n = data_dict["observations"].shape[0]
    indices = jax.random.randint(key, (batch_size,), 0, n)
    return {k: jnp.asarray(v[indices]) for k, v in data_dict.items()}

def get_batches(data_dict, batch_size, N, key, epoch):
    for i in range(N):
        yield _sample(data_dict, jax.random.fold_in(key, i + epoch * N), batch_size)

def main(cfg_filename):
    # Load training config
    with open(cfg_filename, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Initialise datadict
    data_dict = {}

    # Load dataset (assume dataset has been processed)
    with h5py.File(cfg["dataset_filename"], 'r') as f:
        for k in list(f.keys()):
            data_dict[k] = f[k][:]

    # Add additional data
    cfg = resolve_dims(
        cfg,
        data_dict["observations"].shape[-1],
        data_dict["actions"].shape[-1]
    )

    # Create rng key
    data_key, train_key = jax.random.split(jax.random.key(cfg["seed"]))
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
    metrics_log = []
    i = 0
    for epoch in tqdm(range(cfg["epochs"]), desc=f"Epochs", leave=True):
        metrics_log_epoch = []
        for batch in tqdm(get_batches(data_dict, cfg["batch_size"], cfg["num_batches"], data_key, epoch), desc=f"Batches", leave=False):
            i += 1
            obs = batch["observations"]
            next_obs = batch["next_observations"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            safety_costs = batch["costs"]
            terminals = batch["terminals"]
            metrics = world_model.train_step(
                obs, next_obs, actions, rewards, safety_costs, terminals, jax.random.fold_in(train_key, epoch * cfg["num_batches"] + i)
            )
            metrics_log_epoch.append(jax.device_get(metrics))
        metrics_log.append(metrics_log_epoch)

    # Save training metrics
    save_path_metrics = (Path(__file__).parent / f'SaveData/{cfg["data_dir"]}/Metrics').resolve()
    save_path_metrics.mkdir(parents=True, exist_ok=True)
    write_metrics(metrics_log, save_path_metrics / "metrics.csv")

    # Save the trained model
    _, state = nnx.split(world_model)
    checkpointer = ocp.StandardCheckpointer()
    save_path_model = (Path(__file__).parent / f'SaveData/{cfg["data_dir"]}/Model').resolve()
    save_path_model.mkdir(parents=True, exist_ok=True)
    checkpointer.save(save_path_model / 'state', state)
    checkpointer.wait_until_finished()
    return 0

if __name__ == "__main__":
    main("TrainingConfigs/wmTraining/standard_model.yaml")