import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "src/save_data/experiments"

def print_results(experiment, controllers, seeds):
    data = load_data(experiment, controllers, seeds)
    for i in range(len(data)):
        avg_steps = data[controllers[i]]["steps"].mean()
        avg_return = data[controllers[i]]["return"].mean()
        avg_plan_time = data[controllers[i]]["plan_time_s"].mean()
        print(f"Controller: {controllers[i]}, Mean Steps: {avg_steps}, Mean Return: {avg_return}, Mean Plan Time: {avg_plan_time}")

def return_results(experiment, controllers, seeds):
    data = load_data(experiment, controllers, seeds)
    avg_steps = []
    avg_return = []
    avg_plan_time = []
    for i in range(len(data)):
        avg_steps.append(data[controllers[i]]["steps"].mean())
        avg_return.append(data[controllers[i]]["return"].mean())
        avg_plan_time.append(data[controllers[i]]["plan_time_s"].mean())
    return avg_steps, avg_return, avg_plan_time

def load_data(experiment, controllers, seeds):
    data = {}
    for controller in controllers:
        controller_data = pd.read_csv(RESULTS_PATH / experiment / f"{controller}_seeds_{seeds[0]}-{seeds[-1]}.csv")
        data[controller] = controller_data
    return data

if __name__ == "__main__":
    controllers = ["CEM", "MPPI", "SCP"]
    
    print("Lipschitz Bound: 1.2")
    experiment_name = "T1_pointgoal1_c_1_2_seed_0"
    seeds = range(20)
    print_results(experiment_name, controllers, seeds)

    print("Lipschitz Bound: 1.5")
    experiment_name = "T1_pointgoal1_c_1_5_seed_0"
    seeds = range(20)
    print_results(experiment_name, controllers, seeds)

    print("Lipschitz Bound: 1.8")
    experiment_name = "T1_pointgoal1_c_1_8_seed_0"
    seeds = range(20)
    print_results(experiment_name, controllers, seeds)

    print("Lipschitz Bound: inf")
    experiment_name = "T1_pointgoal1_c_inf_seed_0"
    seeds = range(20)
    print_results(experiment_name, controllers, seeds)