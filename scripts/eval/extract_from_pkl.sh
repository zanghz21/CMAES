PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF/CMAES
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

LOGDIR=logs/2024-07-15_16-40-43_competition-highway-33x36-cma-es-warehouse-map-400-agents-cnn-iter-update_M7wQaGVH
python env_search/analysis/load_from_reload.py --logdir=$LOGDIR