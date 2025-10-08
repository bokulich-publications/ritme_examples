#!/bin/bash

#SBATCH --job-name="n5_automl_u3_meta"
#SBATCH -A es_bokulich
#SBATCH --nodes=1
#SBATCH --cpus-per-task=50
#SBATCH --time=119:59:59
#SBATCH --mem-per-cpu=4096
#SBATCH --output="/cluster/project/bokulich/adamova/ritme_usecase_runs_final/logs_automl/%x_out.txt"
#SBATCH --open-mode=append

set -x

python ../src/generic_automl.py \
    --total-time-s 115797 \
    --usecase u3_meta \
    --data-splits-folder u3_mlp_prediction/data_splits_u3_metacardis_log \
    --path-to-features ../data/u3_mlp_nishijima24/metacardis_otu_table.tsv \
    --path-to-md ../data/u3_mlp_nishijima24/md_metacardis.tsv \
    --target count_log10 \
    --single-best \
    --restricted-model gradient_boosting

# "ard_regression", "random_forest", "gradient_boosting", "mlp",

# get elapsed time of job
sstat -j $SLURM_JOB_ID
echo "TIME COUNTER:"
sacct -j $SLURM_JOB_ID --format=elapsed --allocations
