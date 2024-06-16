PROJECT_DIR=/media/project0/hongzhi/TrafficFlowMAPF/CMAES
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# LOGDIR=logs/2024-05-27_19-43-47_trafficmapf-sortation-small-linear_CtCaYD5W
# LOGDIR=logs/2024-05-29_14-18-38_trafficmapf-sortation-small-linear_bpUMd5iC
# LOGDIR=logs/2024-05-29_20-11-34_trafficmapf-sortation-small-linear_VC5e5L7E
# LOGDIR=logs/2024-05-29_23-11-10_trafficmapf-sortation-small-linear_kaeME4XX # small, 800, linear
# LOGDIR=logs/2024-05-30_11-25-49_trafficmapf-sortation-small-linear_m4X73KKj # small, 800, quad
# LOGDIR=logs/2024-05-31_20-27-35_trafficmapf-sortation-small-linear_Y2BNmdhM # small narrow, 800, quad

# LOGDIR=logs/2024-06-10_14-48-36_trafficmapf-sortation-small-linear_HSTzCK7s # small
# LOGDIR=logs/2024-05-31_20-29-22_trafficmapf-sortation-60x100_2y5tPVFv # 60x100, 2400, quad
# LOGDIR=logs/2024-06-08_23-19-35_trafficmapf-sortation-small-linear_G4PbgTSF # +nn cache
# LOGDIR=logs/2024-06-09_11-03-56_trafficmapf-sortation-small-linear_ZPQYcBNa # rerun small narrow, 800, quad
# LOGDIR=logs/2024-06-09_15-45-49_trafficmapf-33x36_gtsCJda3 # GGO
# LOGDIR=logs/2024-06-09_15-48-13_trafficmapf-sortation-small_5SnPespx # random tasks
# LOGDIR=logs/2024-06-09_22-49-43_trafficmapf-sortation-small-narrow_BnKBTpip # random tasks rerun
# LOGDIR=logs/2024-06-10_14-34-15_trafficmapf-sortation-small-narrow_MFGq3duw # random tasks, use nn cache
LOGDIR=logs/2024-06-11_10-41-43_trafficmapf-sortation-small-narrow_htsMJDwh # nn cache, retry
# LOGDIR=logs/2024-06-10_10-39-13_trafficmapf-33x36_DJrApRxd # GGO, sim10
# LOGDIR=logs/2024-06-12_12-15-57_trafficmapf-sortation-small-narrow_h2dFuP5r # nn cache, multiple eval
# LOGDIR=logs/2024-06-12_12-16-39_trafficmapf-sortation-small-narrow_GH3b8NXa # nn cache
LOGDIR=logs/2024-06-13_19-07-26_trafficmapf-sortation-small-narrow_v7e54NWb # win_r=1
LOGDIR=logs/2024-06-14_19-44-27_trafficmapf-win-r_sGdmyEH8 # win_r=1, again
LOGDIR=logs/2024-06-15_10-08-50_trafficmapf-win-r_4rCLKFHQ # win_r=0
python env_search/traffic_mapf/eval.py --logdir=$LOGDIR