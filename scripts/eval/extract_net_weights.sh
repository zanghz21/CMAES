#!/bin/bash

USAGE="Usage: bash scripts/plot_tile_usage.sh LOGDIR MODE DOMAIN"

LOGDIR="logs/2024-05-23_01-08-09_trafficmapf-sortation-small-quad_9JCyiXNf"
LOGDIR_TYPE="qd"
MODE="extreme"
DOMAIN="trafficMAPF"

PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF/CMAES
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
shift 4
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