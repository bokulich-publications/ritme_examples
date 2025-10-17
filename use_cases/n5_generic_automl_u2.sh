#!/bin/bash

#SBATCH --job-name="n5_automl_u2"
#SBATCH -A es_bokulich
#SBATCH --nodes=1
#SBATCH --cpus-per-task=100
#SBATCH --time=119:59:59
#SBATCH --mem-per-cpu=5120
#SBATCH --output="/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/logs_automl/%x_out.txt"
#SBATCH --open-mode=append

set -x

python ../src/generic_automl.py \
    --total-time-s 428400 \
    --usecase u2 \
    --data-splits-folder u2_metagenome_ocean/data_splits_u2 \
    --path-to-features ../data/u2_tara_ocean/otu_table_tara_ocean.tsv \
    --path-to-md ../data/u2_tara_ocean/md_tara_ocean.tsv \
    --target temperature_mean_degc \
    --single-best

# "ard_regression", "random_forest", "gradient_boosting", "mlp",

# get elapsed time of job
sstat -j $SLURM_JOB_ID
echo "TIME COUNTER:"
sacct -j $SLURM_JOB_ID --format=elapsed --allocations
