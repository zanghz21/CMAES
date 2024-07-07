CONFIG=config/traffic_mapf/sortation_small.gin
SEED=1
HPC_CONFIG=config/hpc/run_cpu.sh
bash scripts/run_slurm.sh $CONFIG $SEED $HPC_CONFIG