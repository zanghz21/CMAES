import argparse
import json
import os
from env_search.competition.update_model.utils import Map
from env_search.utils import get_n_valid_edges, get_n_valid_vertices
from env_search.competition.config import CompetitionConfig
from env_search.competition.module import CompetitionModule
from env_search.utils.logging import get_current_time_str

import numpy as np
import gin
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler()          # Log to console
                    ])

def parse_config(log_dir):
    cfg_file = os.path.join(log_dir, "config.gin")
    gin.parse_config_file(cfg_file, skip_unknown=True)
    config = CompetitionConfig()
    return config
    
def get_update_model(log_dir):
    update_model_file = os.path.join(log_dir, "optimal_update_model.json")
    with open(update_model_file, "r") as f:
        model_json = json.load(f)
        
    params = model_json["params"]
    model_params = np.array(params)
    return model_params

def parse_map(map_path):
    comp_map = Map(map_path)
    n_e = get_n_valid_edges(comp_map.graph, True, "competition")
    n_v = get_n_valid_vertices(comp_map.graph, "competition")
    return n_e, n_v

def base_exp(cfg: CompetitionConfig, model_params, log_dir):
    n_e, n_v = parse_map(cfg.map_path)
    eval_logdir = os.path.join(log_dir, "eval") # no use here
    module = CompetitionModule(cfg)
    tp_list = []
    for seed in range(50):
        res, _ = module.evaluate_online_update(model_params, eval_logdir, n_e, n_v, seed)
        tp = res['throughput']
        print(f"seed = {seed}, tp = {tp}")
        tp_list.append(tp)
    print(np.mean(tp_list), np.std(tp_list))
    
    
def transfer_exp(cfg: CompetitionConfig, model_params, log_dir):
    map_shapes = ["33x36", "45x47", "57x58", "69x69", "81x80", "93x91"]
    map_shapes = ["33x36"]
    num_agents_lists = [
        [200, 300, 400, 500, 600], 
        [450, 600, 750, 900, 1050], 
        [800, 1000, 1200, 1400, 1600], 
        [1000, 1300, 1600, 1900, 2200], 
        [1000, 1500, 2000, 2500, 3000], 
        [2000, 2500, 3000, 3500, 4000]
    ]
    eval_logdir = os.path.join(log_dir, "eval") # no use here
    os.makedirs(eval_logdir, exist_ok=True)
    
    time_str = get_current_time_str()
    
    logger = logging.getLogger("ggo")
    file_handler = logging.FileHandler(os.path.join(eval_logdir, f"ggo_{time_str}.log"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info("start new experiments!")
    
    for i, map_shape in enumerate(map_shapes):
        map_path = f"maps/competition/expert_baseline/pibt_warehouse_{map_shape}_w_mode_flow_baseline.json" 
        cfg.map_path = map_path
        n_e, n_v = parse_map(map_path)
        for ag in num_agents_lists[i]:
            cfg.num_agents = ag
            module = CompetitionModule(cfg)
        
            tp_list = []
            for seed in range(5):
                res, _ = module.evaluate_online_update(model_params, eval_logdir, n_e, n_v, seed)
                tp = res['throughput']
                info = f"map={map_path}, ag={ag}, seed = {seed}, tp = {tp}"
                print(info)
                logger.info(info)
                tp_list.append(tp)
            logger.critical(f"map={map_path}, ag={ag}, tp_mean={np.mean(tp_list)}, tp_std={np.std(tp_list)}")
        

def main(log_dir):
    cfg = parse_config(log_dir)
    model_params = get_update_model(log_dir)
    # base_exp(cfg, model_params, log_dir)
    transfer_exp(cfg, model_params, log_dir)
    

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--logdir', type=str, required=True)
    cfg = p.parse_args()
    
    main(log_dir=cfg.logdir)