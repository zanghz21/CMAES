CONFIG=config/competition/online_update/sortation_small_800_dists_time_future.gin
CONFIG=config/competition/online_update/sortation_small_800_dists_time.gin
CONFIG=config/competition/online_update/sortation_small_800_dists_time_no_late.gin
CONFIG=config/competition/cnn_iter_update/sortation_small_800_dists_time.gin
# CONFIG=config/competition/cnn_iter_update/33x36_400_offline_in_online_env.gin
# CONFIG=config/competition/cnn_iter_update/33x36_400_offline_in_online_env_origin.gin

NUM_WORKERS=50
PARTITION=RM
SEED=3
TOTAL_TIME=24:00:00

bash scripts/run_slurm_psc_RM_local.sh $CONFIG $NUM_WORKERS $SEED $PARTITION $TOTAL_TIME