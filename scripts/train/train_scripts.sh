PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF/
# CONFIG=config/traffic_mapf/sortation_small_narrow.gin
# CONFIG=config/traffic_mapf/warehouse_small_narrow.gin
CONFIG=config/traffic_mapf/warehouse_30x120.gin
# CONFIG=config/traffic_mapf/warehouse_60x100.gin
# CONFIG=config/traffic_mapf/sortation_medium.gin
# CONFIG=config/traffic_mapf/ost.gin
# CONFIG=config/traffic_mapf/warehouse.gin
# CONFIG=config/traffic_mapf/ost_downsample.gin

SEED=9
NUM_WORKERS=16
# RELOAD_DIR=logs/2024-06-29_18-03-34_trafficmapf-warehouse-small-narrow_vBu5Aga6

bash scripts/run_local.sh $CONFIG $SEED $NUM_WORKERS -p $PROJECT_DIR
# bash scripts/run_local.sh $CONFIG $SEED $NUM_WORKERS -p $PROJECT_DIR -r $RELOAD_DIR