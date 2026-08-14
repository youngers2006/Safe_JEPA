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
eval "$(conda shell.bash hook)"
conda activate safe_JEPA

export WANDB_API_KEY="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"
export MUJOCO_GL="egl"
CUSPARSE_DIR=$(dirname $(find $CONDA_PREFIX -name libcusparse.so* | head -n 1))
CUBLAS_DIR=$(dirname $(find $CONDA_PREFIX -name libcublas.so* | head -n 1))
export LD_LIBRARY_PATH=$CUSPARSE_DIR:$CUBLAS_DIR:$LD_LIBRARY_PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"
python training_point_maze.py