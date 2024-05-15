import numpy as np

from env_search.utils.worker_state import get_traffic_mapf_module

def run_traffic_mapf(
    nn_weights: np.ndarray, 
    net_type: str
):
    traffic_mapf_module = get_traffic_mapf_module()
    result = traffic_mapf_module.evaluate(
        nn_weights, net_type
    )
    return result

def process_traffic_mapf_results(curr_result_json):
    traffic_mapf_module = get_traffic_mapf_module()
    
    results = traffic_mapf_module.process_eval_result(curr_result_json)
    return results