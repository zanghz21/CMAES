PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF/
CONFIG=config/learn2follow/test.gin

SEED=21
NUM_WORKERS=10

bash scripts/run_local.sh $CONFIG $SEED $NUM_WORKERS -p $PROJECT_DIR
# bash scripts/run_local.sh $CONFIG $SEED $NUM_WORKERS -p $PROJECT_DIR -r $RELOAD_DIR