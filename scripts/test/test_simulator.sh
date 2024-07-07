PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF
SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0 --bind ${PROJECT_DIR}:${PROJECT_DIR}"

singularity exec ${SINGULARITY_OPTS} ../singularity/ubuntu_onlineGGO.sif \
    python env_search/test.py