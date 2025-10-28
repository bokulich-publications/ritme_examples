#!/bin/bash

#SBATCH --job-name="logs_work"
#SBATCH -A es_bokulich
#SBATCH --nodes=1
#SBATCH --cpus-per-task=60
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=5120
#SBATCH --output="%x_out.txt"
#SBATCH --open-mode=append

set -x

python ../src/extract_mlflow_logs.py --dirs \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u1_linreg_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u1_nncorn_tpe3 \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u1_nnreg_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u1_rf_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u1_xgb_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u1_xgb_tpe_restricted \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u1_trac_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u1_nnclass_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u2_trac_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u2_nnclass_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u2_linreg_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u2_linreg_tpe_restricted \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u2_nncorn_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u2_nnreg_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u2_rf_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u2_xgb_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u3_galaxy_log_linreg_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u3_galaxy_log_nnreg_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u3_galaxy_log_rf_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u3_galaxy_log_xgb_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final_time/u3_galaxy_log_xgb_tpe_w_start \
-o merged_all_trials
