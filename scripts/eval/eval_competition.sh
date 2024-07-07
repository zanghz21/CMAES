PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF
SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0 --bind ${PROJECT_DIR}:${PROJECT_DIR}"

LOGDIR="logs/2024-06-29_22-16-55_competition-highway-32x32-cma-es-random-map-400-agents-cnn-iter-update_KsDyKFQs"

singularity exec ${SINGULARITY_OPTS} ../singularity/ubuntu_onlineGGO.sif \
    python env_search/competition/eval.py --logdir=$LOGDIR