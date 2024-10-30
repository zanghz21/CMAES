import os
import pandas as pd
from env_search.traffic_mapf.multi_process_eval import EXP_AGENTS
import argparse

def parse_eval(base_dir, timestr):
    for map_type in EXP_AGENTS:
        if map_type in base_dir:
            if "warehouse" in map_type:
                if "large" in base_dir:
                    map_type = "warehouse_large"
                elif "narrow" in base_dir:
                    map_type = "warehouse_small_narrow"
                else:
                    map_type = "warehouse_small"
            break

    agent_ls = EXP_AGENTS[map_type]
    
    for ag in agent_ls:
        file = os.path.join(base_dir, f"ag{ag}", f"{timestr}.csv")
        if not os.path.exists(file):
            print(f"file [{file}] not found!")
            continue
        df = pd.read_csv(file, sep="\t")
        print(f"base_exp = {base_dir}, ag={ag}, num_exp = {len(df)}, avg tp = {df['tp'].mean()}, std tp = {df['tp'].std()}")
        if "sim_time" in df.keys():
            print(f"avg time = {df['sim_time'].mean()}, std time = {df['sim_time'].std()}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str)
    p.add_argument("--time_str", type=str)
    cfg = p.parse_args()
    
    parse_eval(cfg.base_dir, cfg.time_str)
