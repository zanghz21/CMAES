PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF/CMAES
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

#!/bin/sh


# BASE_JSON_PATH=logs/2024-05-30_11-25-49_trafficmapf-sortation-small-linear_m4X73KKj/eval_json
# AGENT_NUMS="2000 6000 10000 14000 18000"
# for agent_num in $AGENT_NUMS
# do
#     JSON_PATH=$BASE_JSON_PATH/sortation_medium_${agent_num}.json
#     python env_search/traffic_mapf/plot.py --json_path=$JSON_PATH
# done

# AGENT_NUMS="1200 1800 2400 3000 3600"
# for agent_num in $AGENT_NUMS
# do
#     JSON_PATH=$BASE_JSON_PATH/sortation_60x100_${agent_num}.json
#     python env_search/traffic_mapf/plot.py --json_path=$JSON_PATH
# done

# JSON_PATH=logs/2024-07-01_14-42-36_trafficmapf-warehouse-30x120_SghPbuLQ/eval_json/warehouse_30x120_20240701_224712.json
# python env_search/traffic_mapf/plot.py --json_path=$JSON_PATH


JSON_PATH=../Guided-PIBT/baseline_results/warehouse_30x120_1000_3.json
MAP_PATH=/media/project0/hongzhi/TrafficFlowMAPF/Guided-PIBT/guided-pibt/benchmark-lifelong/maps/warehouse_30x120.map
python env_search/traffic_mapf/plot.py --json_path=$JSON_PATH --map_path=$MAP_PATH