PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF
SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0 --bind ${PROJECT_DIR}:${PROJECT_DIR}"

LOGDIR=logs/2024-07-23_13-34-56_32x32-random-400-online-only-task_ghkA8vPm

singularity exec ${SINGULARITY_OPTS} ../singularity/ubuntu_onlineGGO.sif \
    python env_search/competition/eval_to_log.py --logdir=$LOGDIR