# LOGDIR=/ocean/projects/cis220074p/hzang/trafficmapf_log/2024-07-12_16-20-55_trafficmapf-sortation-small_9w8xnTRB
LOGDIR=/ocean/projects/cis220074p/hzang/trafficmapf_log/2024-07-12_16-25-16_trafficmapf-room_S8bkihgT
N_WORKERS=50
N_EVALS=50
ALL_RESULTS_DIR=/ocean/projects/cis220074p/hzang/online/results
PARTITION=RM
TOTAL_TIME=05:00:00

bash scripts/eval/eval_trafficflow_psc.sh $LOGDIR $N_WORKERS $N_EVALS $ALL_RESULTS_DIR $PARTITION $TOTAL_TIME