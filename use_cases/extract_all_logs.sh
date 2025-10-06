#!/bin/bash

#SBATCH --job-name="logs_work"
#SBATCH -A es_bokulich
#SBATCH --nodes=1
#SBATCH --cpus-per-task=50
#SBATCH --time=119:59:59
#SBATCH --mem-per-cpu=5120
#SBATCH --output="%x_out.txt"
#SBATCH --open-mode=append

set -x

python ../src/extract_mlflow_logs.py --dirs \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u1_linreg_random \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u1_linreg_tpe \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u1_nnclass_random \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u1_nnclass_tpe \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u1_nncorn_tpe3 \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u1_nnreg_random \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u1_nnreg_tpe \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u1_rf_random \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u1_rf_tpe \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u1_trac_random \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u1_trac_tpe \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u1_xgb_random \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u1_xgb_tpe \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u2_linreg_random \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u2_linreg_tpe \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u2_nnclass_tpe \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u2_nncorn_tpe \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u2_nnreg_random \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u2_nnreg_tpe \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u2_rf_random \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u2_rf_tpe \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u2_xgb_random \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u2_xgb_tpe \
-o merged_u1_u2_runs_work_no2trac
