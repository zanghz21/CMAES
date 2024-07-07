#!/bin/bash

USAGE="Usage: bash scripts/plot_tile_usage.sh LOGDIR MODE DOMAIN"

LOGDIR="slurm_logs/2024-07-03_11-02-21_trafficmapf-room_eu6K77WM"
LOGDIR_TYPE="qd"
MODE="extreme"
DOMAIN="trafficMAPF"

# LOGDIR="logs/2024-06-29_22-16-55_competition-highway-32x32-cma-es-random-map-400-agents-cnn-iter-update_KsDyKFQs"
# DOMAIN="competition"

PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF/CMAES
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
# shift 4
while getopts "p:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      *) echo "Invalid option. ${USAGE}"
  esac
done


if [ -z "${LOGDIR}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${LOGDIR_TYPE}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${MODE}" ]
then
  echo "${USAGE}"
  exit 1
fi

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:${PROJECT_DIR}"
fi

python env_search/analysis/tile_usage.py \
    --logdir "$LOGDIR" \
    --logdir-type "$LOGDIR_TYPE" \
    --mode "$MODE" \
    --domain "$DOMAIN"