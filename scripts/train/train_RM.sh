CONFIG=config/traffic_mapf/ost.gin
NUM_WORKERS=50
PARTITION=RM-512
SEED=2
TOTAL_TIME=48:00:00

# bash scripts/run_slurm_psc_RM_local.sh $CONFIG $NUM_WORKERS $SEED $PARTITION $TOTAL_TIME
bash scripts/run_slurm_psc_RM_local.sh $CONFIG $NUM_WORKERS $SEED $PARTITION $TOTAL_TIME -r $RELOAD_PATH