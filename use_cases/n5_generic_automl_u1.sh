#!/bin/bash

#SBATCH --job-name="n5_automl_u1"
#SBATCH -A es_bokulich
#SBATCH --nodes=1
#SBATCH --cpus-per-task=50
#SBATCH --time=119:59:59
#SBATCH --mem-per-cpu=5120
#SBATCH --output="/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/logs_automl/%x_out.txt"
#SBATCH --open-mode=append

set -x

python ../src/generic_automl.py \
    --total-time-s 428400 \
    --usecase u1 \
    --data-splits-folder u1_amplicon_age_prediction/data_splits_u1 \
    --path-to-features ../data/u1_subramanian14/otu_table_subr14_wq.tsv \
    --path-to-md ../data/u1_subramanian14/md_subr14.tsv \
    --target age_months \
    --single-best

# "ard_regression", "random_forest", "gradient_boosting", "mlp",

# get elapsed time of job
sstat -j $SLURM_JOB_ID
echo "TIME COUNTER:"
sacct -j $SLURM_JOB_ID --format=elapsed --allocations
