#!/bin/bash
#SBATCH -p sulcgpu3
#SBATCH -q sulcgpu1
#SBATCH -N 1                     # number of GPUs
#SBATCH -n 30                     # number of cores
#SBATCH -t 0-48:00              # wall time (D-HH:MM)
#SBATCH --gres=gpu:5     # number of GPUs
#SBATCH -o slurm.%j.out          # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err          # STDERR (%j = JobId)
#SBATCH --job-name="3hb_unb_0"
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=mlsample@asu.edu     #send to my email

module add cuda/10.2.89                                     # Request Cuda libraries
module add gcc/8.4.0                                        # and the required C compiler \ lib
module add anaconda/py3
module add jupyter/latest

export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe_$SLURM_TASK_PID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log_$SLURM_TASK_PID
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d


python melting_production.py
