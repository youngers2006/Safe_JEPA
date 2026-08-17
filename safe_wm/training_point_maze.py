import ogbench
import numpy as np
import wandb
import flax.nnx as nnx
import flax.serialization
from tqdm import tqdm

from World_Model.WorldModel import WorldModel

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import ogbench
import numpy as np
import wandb
import h5py
import jax
import flax.nnx as nnx
import flax.serialization
from tqdm import tqdm

from World_Model.WorldModel import WorldModel

def get_lazy_batches(h5_file_path: str, batch_size: int, chunk_size: int, key: np.random.Generator):
    """
    Reads data directly from disk in chunks, preventing RAM exhaustion.
    Shuffles data locally within each chunk.
    """
    with h5py.File(h5_file_path, 'r') as f:
        total_steps = f['observations'].shape[0]
        
        # Calculate chunk boundaries
        starts = np.arange(0, total_steps, chunk_size)
        
        # Optionally shuffle the order in which we read chunks
        key.shuffle(starts)
        
        for start_idx in starts:
            end_idx = min(start_idx + chunk_size, total_steps)
            
            # Load only this specific chunk into RAM
            obs_chunk = f['observations'][start_idx:end_idx]
            next_obs_chunk = f['next_observations'][start_idx:end_idx]
            actions_chunk = f['actions'][start_idx:end_idx]
            rewards_chunk = f['rewards'][start_idx:end_idx]
            masks_chunk = f['masks'][start_idx:end_idx]
            
            chunk_len = obs_chunk.shape[0]
            indices = np.arange(chunk_len)
            key.shuffle(indices)
            
            for i in range(0, chunk_len - batch_size + 1, batch_size):
                batch_idx = indices[i:i + batch_size]
                yield (
                    obs_chunk[batch_idx],
                    next_obs_chunk[batch_idx],
                    actions_chunk[batch_idx],
                    rewards_chunk[batch_idx],
                    masks_chunk[batch_idx]
                )

def train():
    Batch_Size = 128
    Epochs = 100
    lr = 3e-4
    tau = 0.01
    seed = 20
    chunk_size = 50000 # Tune this based on your available RAM
    dataset_name = 'visual-cube-single-play-singletask-task1-v0'

    wandb.init(project="scp-jepa-world-model", config={
        "batch_size": Batch_Size, "epochs": Epochs, "lr": lr, "tau": tau, "dataset": dataset_name
    })

    # We only use make_env_and_datasets to get the environment shapes and trigger the download.
    # We DO NOT extract the arrays from the train_dataset dictionary.
    print("Initializing environment and ensuring dataset is downloaded...")
    env, _, _ = ogbench.make_env_and_datasets(dataset_name)

    # You must locate the HDF5 file downloaded by ogbench.
    # It is typically cached in your home directory. 
    # Update this path to match your system's cache location.
    h5_file_path = os.path.expanduser(f"~/.ogbench/{dataset_name}.hdf5")
    
    if not os.path.exists(h5_file_path):
        raise FileNotFoundError(f"Could not find the dataset at {h5_file_path}. Please check where ogbench caches files.")

    # Get total size purely from metadata (Zero RAM cost)
    with h5py.File(h5_file_path, 'r') as f:
        dataset_size = f['observations'].shape[0]
        action_dim = f['actions'].shape[-1]

    obs_shape = env.observation_space.shape
    dynamic_d_in = int(np.prod(obs_shape))

    rngs = nnx.Rngs(seed)
    world_model = WorldModel(
        d_in_obs=dynamic_d_in,
        image_size=64, 
        d_latent=64, 
        d_action=action_dim, 
        lr=lr, 
        rngs=rngs
    )
    np_rng = np.random.default_rng(seed)

    print("Start Training ...")
    for epoch in range(Epochs):
        epoch_metrics = {
            "loss_total": [], "loss_dyn": [], "loss_v": [], 
            "loss_r": [], "loss_safety": [], "loss_var": []
        }

        # Calculate approximate batches per epoch for the tqdm progress bar
        total_batches = dataset_size // Batch_Size
        batch_generator = get_lazy_batches(h5_file_path, Batch_Size, chunk_size, np_rng)

        for batch_data in tqdm(batch_generator, total=total_batches, desc=f"Epoch {epoch+1}/{Epochs}"):
            b_obs_uint, b_next_obs_uint, b_actions, b_rewards, b_masks = batch_data

            # Process data on-the-fly (saves massive amounts of RAM)
            b_obs = (b_obs_uint.astype(np.float32) / 255.0) - 0.5
            b_next_obs = (b_next_obs_uint.astype(np.float32) / 255.0) - 0.5
            b_done = 1.0 - b_masks
            b_safety_costs = (np.max(np.abs(b_actions), axis=-1) > 0.8).astype(np.float32)

            metrics = world_model.train_step(
                b_obs,
                b_next_obs,
                b_actions,
                b_rewards,
                b_safety_costs,
                b_done
            )

            world_model.update_target_networks((tau, tau, tau))

            for k, v in metrics.items():
                epoch_metrics[k].append(v.item())

        avg_metrics = {f"train/{k}": np.mean(v) for k, v in epoch_metrics.items()}
        wandb.log(avg_metrics, step=epoch)

    wandb.finish()
    print("Training complete.")

    print("Extracting and saving model state...")
    model_state = nnx.state(world_model)
    bytes_data = flax.serialization.to_bytes(model_state)
    
    with open("scp_world_model.msgpack", "wb") as f:
        f.write(bytes_data)
        
    print("Model weights successfully saved to scp_world_model.msgpack")

if __name__ == "__main__":
    train()