CONFIG=config/traffic_mapf/warehouse.gin
NUM_WORKERS=20
PARTITION=RM-512
SEED=3
TOTAL_TIME=24:00:00

# RELOAD_PATH=/ocean/projects/cis220074p/hzang/trafficmapf_log/2024-07-03_11-16-08_trafficmapf-game_MF9WJEQd
# RELOAD_PATH=/ocean/projects/cis220074p/hzang/trafficmapf_log/2024-06-28_12-46-36_trafficmapf-warehouse-large_urwCvgHN
RELOAD_PATH=/ocean/projects/cis220074p/hzang/trafficmapf_log/2024-07-07_19-18-00_trafficmapf-warehouse-large_Ya2gXe5k

# bash scripts/run_slurm_psc_RM_local.sh $CONFIG $NUM_WORKERS $SEED $PARTITION $TOTAL_TIME
bash scripts/run_slurm_psc_RM_local.sh $CONFIG $NUM_WORKERS $SEED $PARTITION $TOTAL_TIME -r $RELOAD_PATH