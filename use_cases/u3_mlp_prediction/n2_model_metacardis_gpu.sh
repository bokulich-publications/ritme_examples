#!/bin/bash

#SBATCH --job-name="u3_metacardis_gpu_config"
#SBATCH -A es_bmlbb
#SBATCH --gpus=1
#SBATCH --gres=gpumem:5120m
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=5129
#SBATCH --time=24:00:00
#SBATCH --output="%x_out.txt"
#SBATCH --open-mode=append

module load eth_proxy

set -x

echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_GPUS: $SLURM_GPUS"

# ! USER SETTINGS HERE
# -> config file to use
CONFIG="config/u3_metacardis_gpu_config.json"
# -> path to the metadata file
PATH_MD="../../data/u3_mlp_nishijima24/md_metacardis.tsv"
# -> path to the feature table file
PATH_FT="../../data/u3_mlp_nishijima24/metacardis_otu_table.tsv"
# -> path to taxonomy file
PATH_TAX="../../data/u3_mlp_nishijima24/u3_taxonomy.qza"
# -> path to the .env file
ENV_PATH="../../.env"
# -> path to store model logs
LOGS_DIR="/cluster/work/bokulich/adamova/ritme_usecase_runs"
# -> path to data splits
PATH_DATA_SPLITS="data_splits_u3_metacardis_gpu"

# if your number of threads are limited increase as needed
ulimit -u 60000
ulimit -n 524288
# ! USER END __________

# # Load environment variables from .env
export $(grep -v '^#' "$ENV_PATH" | xargs)

# # CLI version
echo "Running split-train-test"
ritme split-train-test $PATH_DATA_SPLITS $PATH_MD $PATH_FT --train-size 0.8 --seed 12

echo "Running find-best-model-config"
ritme find-best-model-config $CONFIG "${PATH_DATA_SPLITS}/train_val.pkl" --path-to-tax $PATH_TAX --path-store-model-logs $LOGS_DIR

echo "Running evaluate-tuned-models"
# Read the value of "experiment_tag" from the config file
experiment_tag=$(python -c "import json, sys; print(json.load(open('$CONFIG'))['experiment_tag'])")

ritme evaluate-tuned-models "${LOGS_DIR}/${experiment_tag}" "${PATH_DATA_SPLITS}/train_val.pkl" "${PATH_DATA_SPLITS}/test.pkl"

sstat -j $SLURM_JOB_ID

# get elapsed time of job
echo "TIME COUNTER:"
sacct -j $SLURM_JOB_ID --format=elapsed --allocations
