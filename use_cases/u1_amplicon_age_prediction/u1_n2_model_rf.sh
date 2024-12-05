#!/bin/bash

#SBATCH --job-name="test_u1_rf"
#SBATCH -A es_bokulich
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=03:59:59
#SBATCH --mem-per-cpu=2048
#SBATCH --output="%x_out.txt"
#SBATCH --open-mode=append

set -x

echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_GPUS_PER_TASK: $SLURM_GPUS_PER_TASK"

# ! USER SETTINGS HERE
# -> config file to use
CONFIG="q2_ritme/run_config.json"

# if your number of threads are limited increase as needed
ulimit -u 60000
ulimit -n 524288
# ! USER END __________

# Change working directory to the script's directory
cd "$(dirname "$0")"
conda activate ritme_model

python u1_n2_model_rf.py
sstat -j $SLURM_JOB_ID

# get elapsed time of job
echo "TIME COUNTER:"
sacct -j $SLURM_JOB_ID --format=elapsed --allocations
