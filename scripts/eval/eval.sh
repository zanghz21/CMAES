PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF/CMAES
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# LOGDIR=logs/2024-05-27_19-43-47_trafficmapf-sortation-small-linear_CtCaYD5W
# LOGDIR=logs/2024-05-29_14-18-38_trafficmapf-sortation-small-linear_bpUMd5iC
# LOGDIR=logs/2024-05-29_20-11-34_trafficmapf-sortation-small-linear_VC5e5L7E
# LOGDIR=logs/2024-05-29_23-11-10_trafficmapf-sortation-small-linear_kaeME4XX # small, 800, linear
# LOGDIR=logs/2024-05-30_11-25-49_trafficmapf-sortation-small-linear_m4X73KKj # small, 800, quad
# LOGDIR=logs/2024-05-31_20-27-35_trafficmapf-sortation-small-linear_Y2BNmdhM # small narrow, 800, quad
LOGDIR=logs/2024-05-31_20-29-22_trafficmapf-sortation-60x100_2y5tPVFv # 60x100, 2400, quad
python env_search/traffic_mapf/eval.py --logdir=$LOGDIR