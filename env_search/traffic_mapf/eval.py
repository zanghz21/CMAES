import json
import os
import numpy as np
import py_driver
import time
import gin
from env_search.traffic_mapf.config import TrafficMAPFConfig
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
from env_search.traffic_mapf.utils import get_map_name

def get_eval_log_save_path(save_dir, map_path):
    map_name = get_map_name(map_path)
    save_path = os.path.join(save_dir, map_name+".json")
    return save_path

def experiments(base_kwargs, save_dir):
    # t = time.time()
    # print("in py", base_kwargs["save_path"])
    # result_json_s = py_driver.run(**base_kwargs)
    # print("sim_time = ", time.time()-t)
    # print(result_json_s)
    
    
    for ag in [2000, 6000, 10000, 14000, 18000]:
    # for ag in [1200, 1800, 2400, 3000, 3600]:
        #     map_path = f"../Guided-PIBT/guided-pibt/benchmark-lifelong/sortation_60x100_{ag}.json"
        for i in range(1, 6):
            map_path = f"../Guided-PIBT/guided-pibt/benchmark-lifelong/sortation_medium_{i}_{ag}.json"
            
            base_kwargs["map_path"] = map_path
            base_kwargs["save_path"] = get_eval_log_save_path(save_dir, map_path)

            t = time.time()
            result_json_s = py_driver.run(**base_kwargs)
            print("sim_time = ", time.time()-t)
            print(result_json_s)
    
    
def main(log_dir, vis=False):
    net_file = os.path.join(log_dir, "optimal_update_model.json")
    cfg_file = os.path.join(log_dir, "config.gin")
    gin.parse_config_file(cfg_file, skip_unknown=True)

    with open(net_file, "r") as f:
        weights_json = json.load(f)
    network_params = weights_json["params"]
    network_type = weights_json["type"]
    
    if vis:
        vis_param(network_params, net_type=network_type, save_dir=os.path.join(log_dir, 'vis'))

    save_dir = os.path.join(log_dir, "eval_json")
    os.makedirs(save_dir, exist_ok=True)
    
    kwargs = {
        "has_map": gin.query_parameter("TrafficMAPFConfig.has_map"), 
        "has_path": gin.query_parameter("TrafficMAPFConfig.has_path"),
        "has_previous": gin.query_parameter("TrafficMAPFConfig.has_previous"),
        "map_path": gin.query_parameter("TrafficMAPFConfig.map_path"), 
        "net_type": gin.query_parameter("TrafficMAPFConfig.net_type"),
        "output_size": gin.query_parameter("TrafficMAPFConfig.output_size"),  
        # "hidden_size": 20, 
        "simu_time": 1000, 
        "use_all_flow": gin.query_parameter("TrafficMAPFConfig.use_all_flow"), 
        "network_params": json.dumps(network_params), 
        "save_path": get_eval_log_save_path(save_dir, gin.query_parameter("TrafficMAPFConfig.map_path"))
    }
    experiments(kwargs, save_dir)


def vis_param(params, net_type, save_dir):
    directions = ['right', 'down', 'left', 'up']
    colors = [(1, 0, 0, i) for i in np.linspace(0, 1, 100)] 
    red_cmap = LinearSegmentedColormap.from_list('red_cmap', colors, N=100)
    params_arr = np.array(params)
    if net_type == "linear":
        params_arr = params_arr.reshape(4, 4, 5, 5)
        for j, exp_d in enumerate(directions):
            sub_dir = os.path.join(save_dir, f"exp_{exp_d}")
            os.makedirs(sub_dir, exist_ok=True)
            for i, d in enumerate(directions):
                plt.figure(f"{exp_d}_{d}")
                plt.imshow(params_arr[j, i], cmap=red_cmap)
                plt.colorbar()
                plt.title(d)
                plt.savefig(os.path.join(sub_dir, f"{d}.png"))
                plt.close()
    else:
        raise NotImplementedError

    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--logdir', type=str, required=True)
    cfg = p.parse_args()
    
    main(log_dir=cfg.logdir)
    
    