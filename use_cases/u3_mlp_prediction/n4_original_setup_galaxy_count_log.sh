#!/bin/bash

#SBATCH --job-name="n4_original_galaxy_log10"
#SBATCH -A es_bokulich
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=23:59:59
#SBATCH --mem-per-cpu=4096
#SBATCH --output="%x_out.txt"
#SBATCH --open-mode=append

module load eth_proxy

set -x
python n4_original_setup.py galaxy count_log10

# get elapsed time of job
sstat -j $SLURM_JOB_ID
echo "TIME COUNTER:"
sacct -j $SLURM_JOB_ID --format=elapsed --allocations
