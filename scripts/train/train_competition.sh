PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF/
CONFIG=config/competition/online_update/CMA_ES_PIBT_32x32_random-map_400-agents_iter_update.gin

SEED=10
NUM_WORKERS=16
bash scripts/run_local.sh $CONFIG $SEED $NUM_WORKERS -p $PROJECT_DIR