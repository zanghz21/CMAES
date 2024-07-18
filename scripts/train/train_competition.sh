PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF/
CONFIG=config/competition/online_update/33x36_400_has_future.gin
# CONFIG=config/competition/wppl_offline/CMA_ES_PIBT_warehouse-33x36_400-agents_iter_update.gin
# CONFIG=config/competition/online_update/CMA_ES_PIBT_warehouse-33x36_400-agents_iter_update.gin
# CONFIG=config/competition/cnn_iter_update/sortation_small_800.gin
# CONFIG=config/competition/cnn_iter_update/CMA_ES_PIBT_warehouse-33x36_400-agents_iter_update.gin
# CONFIG=config/competition/cnn_iter_update/sortation_small_800_dists_time.gin
# CONFIG=config/competition/online_update/sortation_small_800_dists_time.gin
# CONFIG=config/competition/cnn_iter_update/33x36_400_reset_random.gin
# CONFIG=config/competition/cnn_iter_update/33x36_400_offline_in_online_env.gin
# CONFIG=config/competition/online_update/sortation_small_800_dists_time_future.gin
CONFIG=config/competition/cnn_iter_update/33x36_400_overfit_r.gin



SEED=18
NUM_WORKERS=16

bash scripts/run_local.sh $CONFIG $SEED $NUM_WORKERS -p $PROJECT_DIR
# bash scripts/run_local.sh $CONFIG $SEED $NUM_WORKERS -p $PROJECT_DIR -r $RELOAD_DIR