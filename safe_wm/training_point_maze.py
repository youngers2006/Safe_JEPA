import ogbench
import numpy as np

dataset_name = 'visual-cube-single-play-singletask-task1-v0'

env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)

# Extract the 4D image tensor (N, 64, 64, 3)
obs = train_dataset['observations'] 
next_obs = train_dataset['next_observations']
actions = train_dataset['actions']

# Extract the REAL rewards and fix the done flag
rewards = train_dataset['rewards']
done = 1.0 - train_dataset['masks']

# Dummy safety metric (since OGBench doesn't label safety)
safety_costs = (np.max(np.abs(actions), axis=-1) > 0.8).astype(np.float32)

print(obs.shape)