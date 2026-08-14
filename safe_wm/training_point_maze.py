import ogbench
import numpy as np
import wandb
import flax.nnx as nnx
from tqdm import tqdm

from World_Model import WorldModel

def get_batches(dataset_size, batch_size, key: np.random.Generator):
    indices = np.arange(dataset_size)
    key.shuffle(indices)
    for i in range(0, dataset_size - batch_size + 1, batch_size):
        yield indices[i:i + batch_size]

def train():
    Batch_Size = 128
    Epochs = 100
    lr = 3e-4
    tau = 0.01
    seed = 20
    dataset_name = 'visual-cube-single-play-singletask-task1-v0'

    wandb.init(project="scp-jepa-world-model", config={
        "batch_size": Batch_Size, "epochs": Epochs, "lr": lr, "tau": tau, "dataset": dataset_name
    })

    print("Loading Dataset ...")
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)

    # Extract the 4D image tensor (N, 64, 64, 3)
    obs_uint8 = train_dataset['observations'] 
    next_obs_uint8 = train_dataset['next_observations']
    actions = train_dataset['actions']
    rewards = train_dataset['rewards']
    done = 1.0 - train_dataset['masks']
    safety_costs = (np.max(np.abs(actions), axis=-1) > 0.8).astype(np.float32)

    dataset_size = obs_uint8.shape[0]

    print("Dataset Loaded")

    rngs = nnx.Rngs(seed)
    world_model = WorldModel(
        d_in_obs=3, 
        image_size=64, 
        d_latent=64, 
        d_action=actions.shape[-1], 
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

        for batch_idx in tqdm(get_batches(dataset_size, Batch_Size, np_rng), desc=f"In Epoch {epoch+1} / {Epochs}"):
            b_obs_uint = obs_uint8[batch_idx]
            b_next_obs_uint = next_obs_uint8[batch_idx]
            b_obs = (b_obs_uint.astype(np.float32) / 255.0) - 0.5
            b_next_obs = (b_next_obs_uint.astype(np.float32) / 255.0) - 0.5

            b_actions = actions[batch_idx]
            b_rewards = rewards[batch_idx]
            b_done = done[batch_idx]
            b_safety_costs = safety_costs[batch_idx]

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

if __name__ == "__main__":
    train()