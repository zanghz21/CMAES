#!/bin/bash

PROJECT_DIR=""
SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0 --bind ${PROJECT_DIR}:${PROJECT_DIR}"

SEED=0
LOGDIR=""


singularity exec ${SINGULARITY_OPTS} ../singularity/ubuntu_onlineGGO.sif \
    python env_search/traffic_mapf/vis.py --logdir=$LOGDIR --seed=$SEED --save_files