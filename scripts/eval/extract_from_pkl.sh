PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF/CMAES
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

LOGDIR=logs/2024-06-10_10-39-13_trafficmapf-33x36_DJrApRxd
python env_search/analysis/load_from_reload.py --logdir=$LOGDIR