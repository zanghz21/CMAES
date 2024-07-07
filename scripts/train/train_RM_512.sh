CONFIG=config/traffic_mapf/warehouse.gin
NUM_WORKERS=24
PARTITION=RM-512
SEED=0
# RELOAD_PATH=/ocean/projects/cis220074p/hzang/trafficmapf_log/2024-07-03_11-16-08_trafficmapf-game_MF9WJEQd
# RELOAD_PATH=/ocean/projects/cis220074p/hzang/trafficmapf_log/2024-06-28_12-46-36_trafficmapf-warehouse-large_urwCvgHN

bash scripts/run_slurm_psc_RM_local.sh $CONFIG $NUM_WORKERS $SEED $PARTITION
# bash scripts/run_slurm_psc_RM_local.sh $CONFIG $NUM_WORKERS $SEED $PARTITION -r $RELOAD_PATH