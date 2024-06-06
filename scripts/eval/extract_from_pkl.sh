PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF/CMAES
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

LOGDIR=logs/2024-05-27_11-05-13_trafficmapf-sortation-small-quad_XcK3cpBa
python env_search/analysis/load_from_reload.py --logdir=$LOGDIR