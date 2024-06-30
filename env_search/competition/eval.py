import argparse
import json
import os
from env_search.competition.update_model.utils import Map
from env_search.utils import get_n_valid_edges, get_n_valid_vertices
from env_search.competition.config import CompetitionConfig
from env_search.competition.module import CompetitionModule

import numpy as np
import gin

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
    
def main(log_dir):
    cfg = parse_config(log_dir)
    model_params = get_update_model(log_dir)
    n_e, n_v = parse_map(cfg.map_path)
    eval_logdir = os.path.join(log_dir, "eval") # no use here
    
    module = CompetitionModule(cfg)
    
    for seed in range(10):
        res, _ = module.evaluate_online_update(model_params, eval_logdir, n_e, n_v, seed)
        print(f"seed = {seed}, tp = {res['throughput']}")
    

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--logdir', type=str, required=True)
    cfg = p.parse_args()
    
    main(log_dir=cfg.logdir)