PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF/
SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0 --bind ${PROJECT_DIR}:${PROJECT_DIR} --nv"

CONFIG=config/competition/rl_online/w33x36_400.gin

NUM_WORKERS=32

singularity exec ${SINGULARITY_OPTS} ../singularity/ubuntu_onlineGGO.sif \
    python env_search/train_rl.py --piu_config_file $CONFIG --n_workers=$NUM_WORKERS
