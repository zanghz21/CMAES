import json
import os
import gin.config
import numpy as np
import time
import gin
from env_search.traffic_mapf.config import TrafficMAPFConfig
from env_search.traffic_mapf.module import TrafficMAPFModule
from env_search.traffic_mapf.traffic_mapf_manager import TrafficMAPFManager
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import logging
from env_search.traffic_mapf.utils import get_map_name
from datetime import datetime
import csv
import copy
import multiprocessing
from itertools import repeat

from simulators.trafficMAPF_lns import py_driver as lns_py_driver
from simulators.trafficMAPF import py_driver as base_py_driver
from simulators.trafficMAPF_off import py_driver as off_py_driver

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler()          # Log to console
                    ])
logger = logging.getLogger()

def get_time_str():
    now = datetime.now()
    time_str = now.strftime('%Y%m%d_%H%M%S')
    return time_str


EXP_AGENTS={
    "game": [2000, 4000, 6000, 8000, 10000, 12000], 
    # "room": [100, 500, 1000, 1500, 2000, 2500, 3000],
    "room": [100, 200, 300, 400, 500, 600], 
    "sortation_small": [200, 400, 600, 800, 1000, 1200, 1400],
    "warehouse_large": [2000, 4000, 6000, 8000, 10000, 12000], 
    "ggo33x36": [200, 300, 400, 500, 600, 700, 800], 
    "random": [200, 300, 400, 500, 600, 700, 800]
}

def get_offline_weights(log_dir):
    weights_file = os.path.join(log_dir, "optimal_weights.json")
    with open(weights_file, "r") as f:
        weights_json = json.load(f)
    weights = weights_json["weights"]
    return np.array(weights)


def single_experients(base_kwargs, num_agents, seed, base_save_dir, timestr):
    save_dir = os.path.join(base_save_dir, f"ag{num_agents}", f"{seed}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{timestr}.json")
    
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["seed"] = seed
    kwargs["num_agents"] = num_agents
    simulator = lns_py_driver if kwargs["use_lns"] else base_py_driver
    kwargs["gen_tasks"] = True
    
    t = time.time()
    result_json_s = simulator.run(**kwargs)
    result_json = json.loads(result_json_s)
    sim_time = time.time()-t
    tp = result_json["throughput"]
    
    save_data = {
        "sim_time": sim_time, 
        "throughput": tp
    }
    with open(save_path, 'w') as f:
        json.dump(save_data, f)
    

def single_offline_experiments(base_kwargs, optimal_weights, num_agents, seed, base_save_dir, timestr):
    save_dir = os.path.join(base_save_dir, f"ag{num_agents}", f"{seed}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{timestr}.json")
    
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["seed"] = seed
    kwargs["num_agents"] = num_agents
    simulator = off_py_driver
    kwargs["gen_tasks"] = True
    kwargs["map_weights"] = json.dumps(optimal_weights.flatten().tolist())
    
    t = time.time()
    result_json_s = simulator.run(**kwargs)
    result_json = json.loads(result_json_s)
    sim_time = time.time()-t
    tp = result_json["throughput"]
    
    save_data = {
        "sim_time": sim_time, 
        "throughput": tp
    }
    with open(save_path, 'w') as f:
        json.dump(save_data, f)
    
    
def main(log_dir, n_workers, n_evals, all_results_dir, eval_lns=False):
    log_dir = log_dir[:-1] if log_dir.endswith('/') else log_dir
    net_file = os.path.join(log_dir, "optimal_update_model.json")
    cfg_file = os.path.join(log_dir, "config.gin")
    gin.parse_config_file(cfg_file, skip_unknown=True)
    eval_config = TrafficMAPFConfig()
    eval_module = TrafficMAPFModule(eval_config)

    with open(net_file, "r") as f:
        weights_json = json.load(f)
    network_params = weights_json["params"]

    try:
        offline = gin.query_parameter("TrafficMAPFManager.offline")
    except ValueError:
        offline = False
    
    if offline:
        algo = "offline"
    else:
        if eval_config.use_lns:
            algo = "NN_train_lns"
        elif eval_lns:
            algo = "NN_eval_lns"
        else:
            algo = "NN_no_lns"
    
    kwargs = eval_module.gen_sim_kwargs(nn_weights_list=network_params)
    kwargs["seed"] = 0
    kwargs["use_lns"] = eval_config.use_lns or eval_lns
    
    train_map = kwargs["map_path"]
    if "ost" in train_map:
        exp_name = "game"
    elif "sortation" in train_map:
        exp_name = "sortation_small"
    elif "warehouse" in train_map:
        exp_name = "warehouse_large"
    elif "room" in train_map:
        exp_name = "room"
    elif "33x36" in train_map:
        exp_name = "ggo33x36"
    elif "random" in train_map:
        exp_name = "random"
    else:
        print(train_map)
        raise NotImplementedError
    
    log_name = log_dir.split('/')[-1]
    save_dir = os.path.join(all_results_dir, "Guided-PIBT", algo, exp_name, log_name)
    os.makedirs(save_dir, exist_ok=True)
    
    timestr = get_time_str()
    
    agent_ls = EXP_AGENTS[exp_name]
    
    pool = multiprocessing.Pool(n_workers)
    
    exp_agent_ls = []
    exp_seed_ls = []
    for a in agent_ls:
        for s in range(n_evals):
            exp_agent_ls.append(a)
            exp_seed_ls.append(s)
    
    n_simulations = len(exp_agent_ls)
    if not offline:
        pool.starmap(
            single_experients, 
            zip(
                repeat(kwargs, n_simulations), 
                exp_agent_ls, 
                exp_seed_ls, 
                repeat(save_dir, n_simulations), 
                repeat(timestr, n_simulations)
            )
        )
    else:
        optimal_weights = get_offline_weights(log_dir)
        kwargs = {
            "simu_time": eval_config.simu_time, 
            "map_path": eval_config.map_path, 
            "gen_tasks": eval_config.gen_tasks,  
            "num_agents": eval_config.num_agents, 
            "num_tasks": eval_config.num_tasks, 
            "hidden_size": eval_config.hidden_size, 
            "task_assignment_strategy": eval_config.task_assignment_strategy, 
            "task_dist_change_interval": eval_config.task_dist_change_interval, 
            "task_random_type": eval_config.task_random_type
        }
        pool.starmap(
            single_offline_experiments, 
            zip(
                repeat(kwargs, n_simulations), 
                repeat(optimal_weights, n_simulations), 
                exp_agent_ls, 
                exp_seed_ls, 
                repeat(save_dir, n_simulations), 
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
    
    