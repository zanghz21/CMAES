PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF/
CONFIG=config/competition/online_update/33x36_400_has_future.gin
# CONFIG=config/competition/wppl_offline/CMA_ES_PIBT_warehouse-33x36_400-agents_iter_update.gin
# CONFIG=config/competition/online_update/CMA_ES_PIBT_warehouse-33x36_400-agents_iter_update.gin
# CONFIG=config/competition/online_update/33x36_400_net.gin

SEED=12
NUM_WORKERS=16

RELOAD_DIR=logs/2024-06-30_12-21-46_competition-highway-33x36-cma-es-warehouse-map-400-agents-cnn-iter-update_tAKgGNh8
RELOAD_DIR=logs/2024-06-30_22-52-58_competition-33x36-400-online-net_o48KSjmV

bash scripts/run_local.sh $CONFIG $SEED $NUM_WORKERS -p $PROJECT_DIR
# bash scripts/run_local.sh $CONFIG $SEED $NUM_WORKERS -p $PROJECT_DIR -r $RELOAD_DIR