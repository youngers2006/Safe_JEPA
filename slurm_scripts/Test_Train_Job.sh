#!/bin/bash
#SBATCH -p gpu
#SBATCH -A Research_Project-DeepMind
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --mail-type=END
#SBATCH --mail-user=sy493@exeter.ac.uk
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=8          
#SBATCH --mem=32G                  
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

cd /lustre/home/sy493/Safe_JEPA/safe_wm

module load nvidia-cuda/12.1.1

export WANDB_API_KEY="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"
export MUJOCO_GL="egl"
/lustre/home/sy493/.conda/envs/safe_JEPA/bin/python training_point_maze.py