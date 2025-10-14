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
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u1_nnclass_tpe \
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u1_nncorn_tpe3 \
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
/cluster/work/bokulich/adamova/ritme_usecase_runs_final/u2_trac_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final/u3_galaxy_log_linreg_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final/u3_galaxy_log_nnreg_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final/u3_galaxy_log_rf_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final/u3_galaxy_log_xgb_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final/u3_galaxy_log_xgb_tpe_w_start \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final/u3_metacardis_log_linreg_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final/u3_metacardis_log_nnreg_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final/u3_metacardis_log_rf_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final/u3_metacardis_log_xgb_tpe \
/cluster/project/bokulich/adamova/ritme_usecase_runs_final/u3_metacardis_log_xgb_tpe_w_start \
-o merged_all_trials
