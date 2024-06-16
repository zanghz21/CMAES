PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF/
CONFIG=config/traffic_mapf/win_r.gin
SEED=1
NUM_WORKERS=32
bash scripts/run_local.sh $CONFIG $SEED $NUM_WORKERS -p $PROJECT_DIR