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

JSON_PATH=logs/2024-05-31_20-27-35_trafficmapf-sortation-small-linear_Y2BNmdhM/eval_json/sortation_small_narrow_800.json
python env_search/traffic_mapf/plot.py --json_path=$JSON_PATH


# JSON_PATH=/media/project0/hongzhi/TrafficFlowMAPF/Guided-PIBT/baseline_results/sortation_60x100_wide_1200.json
# MAP_PATH=/media/project0/hongzhi/TrafficFlowMAPF/Guided-PIBT/guided-pibt/benchmark-lifelong/maps/sortation_60x100.map
# python env_search/traffic_mapf/plot.py --json_path=$JSON_PATH --map_path=$MAP_PATH