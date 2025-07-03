#!/bin/bash

#SBATCH --job-name="u1_all_config"
#SBATCH -A es_bokulich
#SBATCH --nodes=1
#SBATCH --cpus-per-task=50
#SBATCH --time=119:59:59
#SBATCH --mem-per-cpu=14336
#SBATCH --output="%x_out.txt"
#SBATCH --open-mode=append

module load eth_proxy

set -x

echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_GPUS: $SLURM_GPUS"

# ! USER SETTINGS HERE
# -> config file to use
CONFIG="config/u1_all_config.json"
# -> path to the metadata file
PATH_MD="../../data/u1_subramanian14/md_subr14.tsv"
# -> path to the feature table file
PATH_FT="../../data/u1_subramanian14/otu_table_subr14_wq.qza"
# -> path to taxonomy file
PATH_TAX="../../data/u1_subramanian14/taxonomy_subr14.qza"
# -> path to phylogeny file
PATH_PHYLO="../../data/u1_subramanian14/fasttree_tree_rooted_subr14.qza"
# -> path to the .env file
ENV_PATH="../../.env"
# -> path to store model logs
LOGS_DIR="/cluster/work/bokulich/adamova/ritme_example_runs/u1_all_best_model"
# -> path to data splits
PATH_DATA_SPLITS="data_splits_u1"
# -> group columns for train-test split
GROUP_BY_COLUMN="host_id"

# if your number of threads are limited increase as needed
ulimit -u 60000
ulimit -n 524288
# ! USER END __________

# # Load environment variables from .env
export $(grep -v '^#' "$ENV_PATH" | xargs)

# # Python API version
# python u1_n2_model_rf.py

# # CLI version
echo "Running split-train-test"
ritme split-train-test $PATH_DATA_SPLITS $PATH_MD $PATH_FT --group-by-column $GROUP_BY_COLUMN --train-size 0.8 --seed 12

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
