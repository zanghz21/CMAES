PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF
SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0 --bind ${PROJECT_DIR}:${PROJECT_DIR}"

LOGDIR=slurm_logs/2024-07-03_11-16-08_trafficmapf-game_MF9WJEQd
# LOGDIR=slurm_logs/2024-07-03_11-02-21_trafficmapf-room_eu6K77WM

singularity exec ${SINGULARITY_OPTS} ../singularity/ubuntu_onlineGGO.sif \
    python env_search/traffic_mapf/eval_to_log.py --logdir=$LOGDIR