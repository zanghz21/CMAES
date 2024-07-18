PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF
SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0 --bind ${PROJECT_DIR}:${PROJECT_DIR}"

LOGDIR=logs/2024-07-15_16-40-43_competition-highway-33x36-cma-es-warehouse-map-400-agents-cnn-iter-update_M7wQaGVH

singularity exec ${SINGULARITY_OPTS} ../singularity/ubuntu_onlineGGO.sif \
    python env_search/competition/eval.py --logdir=$LOGDIR