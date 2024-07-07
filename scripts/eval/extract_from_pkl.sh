PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF/CMAES
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

LOGDIR=logs/2024-07-02_15-14-38_trafficmapf-warehouse-30x120_Brsh8DtC
python env_search/analysis/load_from_reload.py --logdir=$LOGDIR