import json
import os
import numpy as np
import py_driver
import time
import gin
from env_search.traffic_mapf.config import TrafficMAPFConfig

def main():
    log_dir = 'logs/2024-05-23_01-08-09_trafficmapf-sortation-small-quad_9JCyiXNf'
    net_file = os.path.join(log_dir, "optimal_update_model.json")
    cfg_file = os.path.join(log_dir, "config.gin")
    gin.parse_config_file(cfg_file, skip_unknown=True)

    with open(net_file, "r") as f:
        weights_json = json.load(f)
    network_params = weights_json["params"]

    map_path = "guided-pibt/benchmark-lifelong/sortation_small_0_600.json"
    kwargs = {
        "has_map": gin.query_parameter("TrafficMAPFConfig.has_map"), 
        "has_path": gin.query_parameter("TrafficMAPFConfig.has_path"),
        "has_previous": gin.query_parameter("TrafficMAPFConfig.has_previous"),
        "map_path": gin.query_parameter("TrafficMAPFConfig.map_path"), 
        "net_type": gin.query_parameter("TrafficMAPFConfig.net_type"),
        "output_size": gin.query_parameter("TrafficMAPFConfig.output_size"),  
        # "hidden_size": 20, 
        "simu_time": 200, 
        "use_all_flow": gin.query_parameter("TrafficMAPFConfig.use_all_flow"), 
        "network_params": json.dumps(network_params)
    }

    t = time.time()
    result_json_s = py_driver.run(**kwargs)
    print("sim_time = ", time.time()-t)
    print(result_json_s)

if __name__ == "__main__":
    main()