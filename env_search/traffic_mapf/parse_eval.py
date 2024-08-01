import os
import pandas as pd
from env_search.traffic_mapf.multi_process_eval import EXP_AGENTS

def parse_eval(base_dir, timestr):
    for map_type in EXP_AGENTS:
        if map_type in base_dir:
            break

    agent_ls = EXP_AGENTS[map_type]
    
    for ag in agent_ls:
        file = os.path.join(base_dir, f"ag{ag}", f"{timestr}.csv")
        if not os.path.exists(file):
            print(f"file [{file}] not found!")
            continue
        df = pd.read_csv(file, sep="\t")
        print(f"base_exp = {base_dir}, ag={ag}, num_exp = {len(df)}, avg tp = {df['tp'].mean()}, std tp = {df['tp'].std()}")
        del df

if __name__ == "__main__":
    base_dir = "/ocean/projects/cis220074p/hzang/online/results/Guided-PIBT/NN_train_lns/ggo33x36/2024-07-26_14-26-33_trafficmapf-33x36_dGrsRyr3/logs"
    time_str = "20240728_115042"
    parse_eval(base_dir, time_str)