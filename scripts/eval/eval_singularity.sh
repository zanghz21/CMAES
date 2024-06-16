PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF
SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0 --bind ${PROJECT_DIR}:${PROJECT_DIR}"

LOGDIR=logs/2024-06-14_19-44-27_trafficmapf-win-r_sGdmyEH8 # win_r=1, again

singularity exec ${SINGULARITY_OPTS} ../singularity/ubuntu_warehouse.sif \
    python env_search/traffic_mapf/eval.py --logdir=$LOGDIR