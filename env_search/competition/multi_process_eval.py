import argparse
import json
import os
from env_search.competition.update_model.utils import Map
from env_search.utils import get_n_valid_edges, get_n_valid_vertices
from env_search.competition.config import CompetitionConfig
from env_search.competition.module import CompetitionModule
from env_search.competition.competition_manager import CompetitionManager
from env_search.iterative_update.envs.online_env_new import CompetitionOnlineEnvNew
from env_search.utils.logging import get_current_time_str

import numpy as np
import gin
import logging
import csv
import copy
import multiprocessing
from itertools import repeat


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler()          # Log to console
                    ])

EXP_AGENTS = {
    "sortation_small": [200, 400, 600, 800, 1000, 1200, 1400],
    "ggo33x36": [200, 300, 400, 500, 600, 700, 800], 
    "random": [200, 300, 400, 500, 600, 700, 800], 
    "room": [100, 500, 1000, 1500, 2000, 2500, 3000]
}
def parse_config(log_dir):
    cfg_file = os.path.join(log_dir, "config.gin")
    gin.parse_config_file(cfg_file, skip_unknown=True)
    config = CompetitionConfig()
    try:
        is_online = gin.query_parameter("CompetitionManager.online_update")
    except ValueError:
        is_online = False
    return config, is_online
    
def get_update_model(log_dir):
    update_model_file = os.path.join(log_dir, "optimal_update_model.json")
    with open(update_model_file, "r") as f:
        model_json = json.load(f)
        
    params = model_json["params"]
    model_params = np.array(params)
    return model_params

def get_offline_weights(log_dir):
    weights_file = os.path.join(log_dir, "optimal_weights.json")
    with open(weights_file, "r") as f:
        weights_json = json.load(f)
    weights = weights_json["weights"]
    return np.array(weights)

def parse_map(map_path):
    comp_map = Map(map_path)
    n_e = get_n_valid_edges(comp_map.graph, True, "competition")
    n_v = get_n_valid_vertices(comp_map.graph, "competition")
    return n_e, n_v


def single_offline_exp(log_dir, optimal_weights, n_e, n_v, seed, base_save_dir, num_agents, timestr):
    cfg, _ = parse_config(log_dir)
    
    save_dir = os.path.join(base_save_dir, f"ag{num_agents}", f"{seed}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{timestr}.json")
    
    cfg_ = copy.deepcopy(cfg)
    cfg_.num_agents = num_agents
    
    
    env = CompetitionOnlineEnvNew(n_v, n_e, cfg, seed=seed)
    obs, info = env.reset()
    done = False
    while not done:
        obs, rew, terminated, truncated, info = env.step(optimal_weights)
        done = terminated or truncated
    tp = info["result"]["throughput"]
    
    save_data = {"throughput": tp}
    with open(save_path, 'w') as f:
        json.dump(save_data, f)
    
    
def single_online_exp(log_dir, model_params, n_e, n_v, seed, base_save_dir, num_agents, timestr):
    cfg, _ = parse_config(log_dir)
    
    save_dir = os.path.join(base_save_dir, f"ag{num_agents}", f"{seed}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{timestr}.json")
    
    cfg_ = copy.deepcopy(cfg)
    cfg_.num_agents = num_agents
    
    module = CompetitionModule(cfg_)
    res, _ = module.evaluate_online_update(model_params, "", n_e, n_v, seed, env_type="new")
    tp = res['throughput']
    
    save_data = {"throughput": tp}
    with open(save_path, 'w') as f:
        json.dump(save_data, f)


def main(log_dir, n_workers, n_evals, all_results_dir):
    base_save_dir = os.path.join(all_results_dir, "PIBT")
    timestr = get_current_time_str()
    
    log_dir = log_dir[:-1] if log_dir.endswith('/') else log_dir
    log_name = log_dir.split('/')[-1]
    
    cfg, is_online = parse_config(log_dir)
    print("is_online:", is_online)
    
    n_e, n_v = parse_map(cfg.map_path)
    
    if "sortation" in cfg.map_path:
        map_type = "sortation_small"
    elif "33x36" in cfg.map_path:
        map_type = "ggo33x36"
    elif "random" in cfg.map_path:
        map_type = "random"
    elif "room" in cfg.map_path:
        map_type = "room"
    else:
        print(f"map path [{cfg.map_path}] cannot be recognized")
        raise NotImplementedError
    
    agent_ls = EXP_AGENTS[map_type]    
    pool = multiprocessing.Pool(n_workers)
    
    exp_agent_ls = []
    exp_seed_ls = []
    for a in agent_ls:
        for s in range(n_evals):
            exp_agent_ls.append(a)
            exp_seed_ls.append(s)
    n_simulations = len(exp_agent_ls)
    
    if is_online:
        algo = "online"
        model_params = get_update_model(log_dir)
        save_dir = os.path.join(base_save_dir, algo, map_type, log_name)
        pool.starmap(
            single_online_exp, 
            zip(
                repeat(log_dir, n_simulations), 
                repeat(model_params, n_simulations), 
                repeat(n_e, n_simulations), 
                repeat(n_v, n_simulations), 
                exp_seed_ls, 
                repeat(save_dir, n_simulations), 
                exp_agent_ls, 
                repeat(timestr, n_simulations)
            )
        )
    else:
        algo = "offline"
        optimal_weights = get_offline_weights(log_dir)
        
        # offline modifications on cfg
        cfg.h_update_late = False
        cfg.past_traffic_interval = 1000
        cfg.simulation_time = 1000
        cfg.update_interval = 1000
        cfg.warmup_time = 100
    
        save_dir = os.path.join(base_save_dir, algo, map_type, log_name)
        pool.starmap(
            single_offline_exp, 
            zip(
                repeat(log_dir, n_simulations),
                repeat(optimal_weights, n_simulations), 
                repeat(n_e, n_simulations), 
                repeat(n_v, n_simulations), 
                exp_seed_ls, 
                repeat(save_dir, n_simulations), 
                exp_agent_ls, 
                repeat(timestr, n_simulations)                
            )
        )
    

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--logdir', type=str, required=True)
    p.add_argument('--n_workers', type=int, required=True)
    p.add_argument('--n_evals', type=int, required=True)
    p.add_argument('--all_results_dir', type=str, default="../results")
    cfg = p.parse_args()
    
    main(log_dir=cfg.logdir, n_workers=cfg.n_workers, n_evals=cfg.n_evals, all_results_dir=cfg.all_results_dir)