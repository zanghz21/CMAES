import json
import os
import gin.config
import numpy as np
import py_driver
import time
import gin
from env_search.traffic_mapf.config import TrafficMAPFConfig
from env_search.traffic_mapf.module import TrafficMAPFModule
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import logging
from env_search.traffic_mapf.utils import get_map_name

def get_eval_log_save_path(save_dir, map_path, suffix=None):
    map_name = get_map_name(map_path)
    filename = map_name if suffix is None else map_name + "_" + suffix
    save_path = os.path.join(save_dir, filename+".json")
    return save_path

def ggo_experiments(base_kwargs, save_dir):
    map_shapes = ["33x36", "45x47", "57x58", "69x69", "81x80", "93x91"]
    num_agents_lists = [
        [200, 300, 400, 500, 600], 
        [450, 600, 750, 900, 1050], 
        [800, 1000, 1200, 1400, 1600], 
        [1000, 1300, 1600, 1900, 2200], 
        [1000, 1500, 2000, 2500, 3000], 
        [2000, 2500, 3000, 3500, 4000]
    ]
    
    logger = logging.getLogger("ggo")
    file_handler = logging.FileHandler('ggo.log')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.critical("start new experiments!")
    
    # logs = []
    for i, map_shape in enumerate(map_shapes):
        map_path = f"../Guided-PIBT/guided-pibt/benchmark-lifelong/maps/ggo_maps/{map_shape}.map"
        base_kwargs["map_path"] = map_path
        for a in num_agents_lists[i]:
            base_kwargs["num_agents"] = a
            base_kwargs["save_path"] = get_eval_log_save_path(save_dir, map_path, str(a))
            throughputs = []
            for j in range(50):
                result_json_s = py_driver.run(**base_kwargs)
                result_json = json.loads(result_json_s)
                throughput = result_json["throughput"]
                throughputs.append(throughput)
                print(f"{map_shape}: agent_num {a}, exp {j}, throughput = {throughput}")
            log = f"{map_shape}: agent_num {a}, avg throughput = {np.mean(throughputs)}, std: {np.std(throughputs)}"
            # print(log)
            logger.critical(log)
            # raise NotImplementedError
            # logs.append(log)
    print("End GGO experiments")
    # for log in logs:
    #     print(log)

def sortation_medium_experiemnts(base_kwargs, save_dir, save_suffix):
    for ag in [2000, 6000, 10000, 14000, 18000]:
    # # # for ag in [1200, 1800, 2400, 3000, 3600]:
    # #     #     map_path = f"../Guided-PIBT/guided-pibt/benchmark-lifelong/sortation_60x100_{ag}.json"
        for i in range(1, 6):
            all_json_path = f"../Guided-PIBT/guided-pibt/benchmark-lifelong/sortation_medium_{i}_{ag}.json"
            
            base_kwargs["all_json_path"] = all_json_path
            base_kwargs["gen_tasks"] = False
            base_kwargs["save_path"] = get_eval_log_save_path(save_dir, all_json_path, suffix=save_suffix)

            t = time.time()
            result_json_s = py_driver.run(**base_kwargs)
            print("sim_time = ", time.time()-t)
            print(result_json_s)
    
    
def experiments(base_kwargs, save_dir, save_suffix):
    t = time.time()
    print("in py", base_kwargs["save_path"])
    # base_kwargs["use_net_cache"] = False
    # base_kwargs["gen_tasks"] = True
    # base_kwargs["num_agents"] = 800
    # base_kwargs["map_path"] = "../Guided-PIBT/guided-pibt/benchmark-lifelong/maps/sortation_small_narrow.map"
    # base_kwargs["num_tasks"] = 100000
    # base_kwargs["seed"] = np.random.randint(1000)
    # for i in range(10):
    #     result_json_s = py_driver.run(**base_kwargs)
    #     # print("sim_time = ", time.time()-t)
    #     print(result_json_s)
    
    # ggo_experiments(base_kwargs, save_dir)
    sortation_medium_experiemnts(base_kwargs, save_dir, save_suffix)
    
    
def main(log_dir, vis=False, suffix="gen_t"):
    net_file = os.path.join(log_dir, "optimal_update_model.json")
    cfg_file = os.path.join(log_dir, "config.gin")
    gin.parse_config_file(cfg_file, skip_unknown=True)
    eval_config = TrafficMAPFConfig()
    eval_module = TrafficMAPFModule(eval_config)

    with open(net_file, "r") as f:
        weights_json = json.load(f)
    network_params = weights_json["params"]
    network_type = weights_json["type"]
    
    if vis:
        vis_param(network_params, net_type=network_type, save_dir=os.path.join(log_dir, 'vis'))

    save_dir = os.path.join(log_dir, "eval_json")
    os.makedirs(save_dir, exist_ok=True)
    
    kwargs = eval_module.gen_sim_kwargs(nn_weights_list=network_params)
    kwargs["seed"] = 0
    save_path = get_eval_log_save_path(save_dir, eval_config.all_json_path, suffix=suffix) if not eval_config.gen_tasks \
        else get_eval_log_save_path(save_dir, eval_config.map_path, suffix=suffix)
    kwargs["save_path"] = save_path
    experiments(kwargs, save_dir, suffix)


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
    
    