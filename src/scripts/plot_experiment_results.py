import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "src/save_data/experiments"

def compare_results(experiment, controllers, seeds):
    data = load_data(experiment, controllers, seeds)
    for i in range(len(data)):
        controller_data = data[i]
        avg_steps = controller_data["steps"].mean()
        avg_return = controller_data["return"].mean()
        avg_plan_time = controller_data["plan_time_s"].mean()
        print(f"Controller: {controllers[i]}, Mean Steps: {avg_steps}, Mean Return: {avg_return}, Mean Plan Time: {avg_plan_time}")

def load_data(experiment, controllers, seeds):
    data = {}
    for controller in controllers:
        controller_data = pd.read_csv(RESULTS_PATH / experiment / f"{controller}_seeds_{seeds[0]}-{seeds[-1]}.csv")
        data[controller] = controller_data
    return data

if __name__ == "__main__":
    experiment_name = "T1_pointgoal1_c_inf_seed_0"
    controllers = ["CEM", "MPPI", "SCP"]
    seeds = range(20)
    compare_results(experiment_name, controllers, seeds)