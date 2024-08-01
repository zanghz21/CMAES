SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"

singularity exec ${SINGULARITY_OPTS} ../singularity/ubuntu_onlineGGO.sif \
    python env_search/competition/parse_eval.py