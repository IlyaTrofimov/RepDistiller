#!/bin/bash
#
#SBATCH --job-name=nas-shufflenetv2
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_small
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

module load gpu/cuda-10.1
module load python/anaconda3

python3.7 train_student.py $@
