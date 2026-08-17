import sys
from unittest.mock import MagicMock

mock = MagicMock()
sys.modules['gymnasium'] = mock
sys.modules['mujoco'] = mock
sys.modules['glfw'] = mock

import ogbench

dataset_name = 'visual-cube-single-play-singletask-task1-v0'

print("Bypassing graphics drivers and initiating download...")
# keys_to_load=[] guarantees it downloads the file but loads 0 bytes into RAM
ogbench.make_env_and_datasets(dataset_name, keys_to_load=[])

print("Download successful! The dataset is cached.")