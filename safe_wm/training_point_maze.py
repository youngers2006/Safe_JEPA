import os
import glob
import numpy as np
import wandb
import h5py
import jax
import flax.nnx as nnx
import flax.serialization
from tqdm import tqdm

# FORBID JAX from hoarding GPU VRAM (Must be at the absolute top)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from World_Model.WorldModel import WorldModel

def get_lazy_batches(h5_file_path: str, batch_size: int, chunk_size: int, key: np.random.Generator):
    """
    Reads data strictly from disk in chunks. RAM usage remains < 1GB.
    """
    with h5py.File(h5_file_path, 'r') as f:
        total_steps = f['observations'].shape[0]
        starts = np.arange(0, total_steps, chunk_size)
        key.shuffle(starts)
        
        for start_idx in starts:
            end_idx = min(start_idx + chunk_size, total_steps)
            
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
    chunk_size = 50000 
    dataset_name = 'visual-cube-single-play-singletask-task1-v0'

    wandb.init(project="scp-jepa-world-model", config={
        "batch_size": Batch_Size, "epochs": Epochs, "lr": lr, "tau": tau, "dataset": dataset_name
    })

    # LOCATE THE DATASET MANUALLY
    data_dir = os.path.expanduser("~/.ogbench/data")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found. Run download.py first.")
        
    hdf5_files = glob.glob(os.path.join(data_dir, f"*{dataset_name}*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"Dataset {dataset_name} not found. Run download.py first.")
        
    h5_file_path = hdf5_files[0]
    print(f"Loading data securely from: {h5_file_path}")
    
    # EXTRACT DIMENSIONS DIRECTLY FROM HDF5 METADATA
    with h5py.File(h5_file_path, 'r') as f:
        dataset_size = f['observations'].shape[0]
        action_dim = f['actions'].shape[-1]
        
    # Hardcoded input dimension for OGBench visual tasks (64x64x3)
    dynamic_d_in = 64 * 64 * 3

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
    total_batches = dataset_size // Batch_Size
    
    for epoch in range(Epochs):
        epoch_metrics = {
            "loss_total": [], "loss_dyn": [], "loss_v": [], 
            "loss_r": [], "loss_safety": [], "loss_var": []
        }

        batch_generator = get_lazy_batches(h5_file_path, Batch_Size, chunk_size, np_rng)

        for batch_data in tqdm(batch_generator, total=total_batches, desc=f"Epoch {epoch+1}/{Epochs}"):
            b_obs_uint, b_next_obs_uint, b_actions, b_rewards, b_masks = batch_data

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

    model_state = nnx.state(world_model)
    with open("scp_world_model.msgpack", "wb") as f:
        f.write(flax.serialization.to_bytes(model_state))
        
if __name__ == "__main__":
    train()