PROJECT_DIR=/jet/home/hzang/TrafficFlowMAPF
SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0 --bind ${PROJECT_DIR}:${PROJECT_DIR}"

# LOGDIR=logs/2024-06-14_19-44-27_trafficmapf-win-r_sGdmyEH8 # win_r=1, again
LOGDIR=logs/2024-06-17_11-58-49_trafficmapf-ost-downsample_sF6q6VHH # game small
LOGDIR=logs/2024-06-18_15-07-29_trafficmapf_p9YBaWqS # game large
LOGDIR=logs/2024-06-17_17-13-21_trafficmapf-sortation-medium_N45EBo4y # sortation medium
LOGDIR=logs/2024-06-27_12-25-55_trafficmapf-warehouse-small-narrow_pBfRwgWL # warehouse small narrow, r=2
LOGDIR=logs/2024-06-27_12-13-07_trafficmapf-warehouse-small-narrow_cAdR2Wd2 # warehouse small narrow, r=3
LOGDIR=logs/2024-06-27_18-50-41_trafficmapf-warehouse-60x100_qcxyuxME # win_r=1
LOGDIR=logs/2024-06-27_19-02-02_trafficmapf-warehouse-small-narrow_MertQ2Mn # win_r=3, longer
LOGDIR=logs/2024-06-29_18-03-34_trafficmapf-warehouse-small-narrow_vBu5Aga6 # win_r=2, longer
LOGDIR=logs/2024-06-29_18-03-20_trafficmapf-warehouse-60x100_tuimeGmk # 60x100, win_r=2
LOGDIR=logs/2024-06-30_17-50-51_trafficmapf-warehouse-small-narrow_pzknu6cL # 30x120, win_r=2
LOGDIR=logs/2024-07-02_15-14-38_trafficmapf-warehouse-30x120_Brsh8DtC

singularity exec ${SINGULARITY_OPTS} ../singularity/ubuntu_onlineGGO.sif \
    python env_search/traffic_mapf/eval.py --logdir=$LOGDIR