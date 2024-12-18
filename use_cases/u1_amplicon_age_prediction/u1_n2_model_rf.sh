#!/bin/bash

#SBATCH --job-name="test_u1_rf_mlflow"
#SBATCH -A share_name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=03:59:59
#SBATCH --mem-per-cpu=2048
#SBATCH --output="%x_out.txt"
#SBATCH --open-mode=append

module load eth_proxy

set -x

echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_GPUS_PER_TASK: $SLURM_GPUS_PER_TASK"

# ! USER SETTINGS HERE
# -> config file to use
CONFIG="u1_rf_config.json"
# Absolute path to the .env file
ENV_PATH="../../.env"

# if your number of threads are limited increase as needed
ulimit -u 60000
ulimit -n 524288
# ! USER END __________

# # Load environment variables from .env
export $(grep -v '^#' "$ENV_PATH" | xargs)

python u1_n2_model_rf.py
sstat -j $SLURM_JOB_ID

# get elapsed time of job
echo "TIME COUNTER:"
sacct -j $SLURM_JOB_ID --format=elapsed --allocations
