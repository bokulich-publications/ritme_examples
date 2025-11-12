#!/bin/bash

#SBATCH --job-name="u2_rf_tpe"
#SBATCH -A SLURM_SHARE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --time=119:59:59
#SBATCH --mem-per-cpu=4096
#SBATCH --output="/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/logs/%x_out.txt"
#SBATCH --open-mode=append

module load eth_proxy

set -x

echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_GPUS: $SLURM_GPUS"

# ! USER SETTINGS HERE
# -> config file to use
CONFIG="config/u2_rf_tpe.json"
# -> path to the metadata file
PATH_MD="../../data/u2_tara_ocean/md_tara_ocean.tsv"
# -> path to the feature table file
PATH_FT="../../data/u2_tara_ocean/otu_table_tara_ocean.tsv"
# -> path to taxonomy file
PATH_TAX="../../data/u2_tara_ocean/taxonomy_tara_ocean.qza"
# -> path to phylogeny file
PATH_PHYLO="../../data/u2_tara_ocean/fasttree_tree_rooted_proc_suna15.qza"
# -> path to the .env file
ENV_PATH="../../.env"
# -> path to store model logs
LOGS_DIR="/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time"
# -> path to data splits
PATH_DATA_SPLITS="data_splits_u2"

# if your number of threads are limited increase as needed
ulimit -u 60000
ulimit -n 524288
# ! USER END __________

# # Load environment variables from .env
export $(grep -v '^#' "$ENV_PATH" | xargs)

# # CLI version
if [[ -f "${PATH_DATA_SPLITS}/train_val.pkl" && -f "${PATH_DATA_SPLITS}/test.pkl" ]]; then
    echo "train_val.pkl and test.pkl already exist. Skipping split-train-test."
else
    echo "Running split-train-test"
    mkdir -p "$PATH_DATA_SPLITS"
    ritme split-train-test "$PATH_DATA_SPLITS" "$PATH_MD" "$PATH_FT" --train-size 0.8 --seed 12
fi

echo "Running find-best-model-config"
ritme find-best-model-config $CONFIG "${PATH_DATA_SPLITS}/train_val.pkl" --path-to-tax $PATH_TAX --path-to-tree-phylo $PATH_PHYLO --path-store-model-logs $LOGS_DIR

echo "Running evaluate-tuned-models"
# Read the value of "experiment_tag" from the config file
experiment_tag=$(python -c "import json, sys; print(json.load(open('$CONFIG'))['experiment_tag'])")

ritme evaluate-tuned-models "${LOGS_DIR}/${experiment_tag}" "${PATH_DATA_SPLITS}/train_val.pkl" "${PATH_DATA_SPLITS}/test.pkl"

sstat -j $SLURM_JOB_ID

# get elapsed time of job
echo "TIME COUNTER:"
sacct -j $SLURM_JOB_ID --format=elapsed --allocations
