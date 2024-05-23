import numpy as np
import json
import gc
import warnings
import subprocess
import os
import hashlib
import time

from env_search.traffic_mapf.config import TrafficMAPFConfig
from env_search.traffic_mapf.result import TrafficMAPFResult
from env_search.utils import MIN_SCORE, get_project_dir
from env_search.utils.logging import get_current_time_str

def generate_hash_file_path():
    file_dir = os.path.join(get_project_dir(), 'run_files')
    os.makedirs(file_dir, exist_ok=True)
    hash_obj = hashlib.sha256()
    raw_name = get_current_time_str().encode() + os.urandom(16)
    hash_obj.update(raw_name)
    file_name = hash_obj.hexdigest()
    file_path = os.path.join(file_dir, file_name)
    return file_path
    
class TrafficMAPFModule:
    def __init__(self, config: TrafficMAPFConfig):
        self.config = config
        
    def evaluate(self, nn_weights: np.ndarray, save_in_disk=True):
        nn_weights_list=nn_weights.tolist()
        
        kwargs = {
            "simu_time": self.config.simu_time, 
            "map_path": self.config.map_path, 
            "has_map": self.config.has_map, 
            "has_path": self.config.has_path, 
            "has_previous": self.config.has_previous,
            "use_all_flow": self.config.use_all_flow, 
            "output_size": self.config.output_size, 
            "net_type": self.config.net_type, 
            "network_params": json.dumps(nn_weights_list)
        }
        delimiter = "[=====delimiter======]"
        if save_in_disk:
            file_path = generate_hash_file_path()
            with open(file_path, 'w') as f:
                json.dump(kwargs, f)
            output = subprocess.run(
                    [
                        'python', '-c', f"""\
import numpy as np
import py_driver
import json
import time

file_path='{file_path}'
with open(file_path, 'r') as f:
    kwargs_ = json.load(f)

one_sim_result_jsonstr = py_driver.run(**kwargs_)
result_json = json.loads(one_sim_result_jsonstr)

print("{delimiter}")
print(result_json)
print("{delimiter}")

                """], stdout=subprocess.PIPE).stdout.decode('utf-8')
            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                raise NotImplementedError
        else:
            output = subprocess.run(['python', '-c', f"""\
import numpy as np
import py_driver
import json

kwargs_ = {kwargs}
one_sim_result_jsonstr = py_driver.run(**kwargs_)
result_json = json.loads(one_sim_result_jsonstr)
np.set_printoptions(threshold=np.inf)
print("{delimiter}")
print(result_json)
print("{delimiter}")
            """], stdout=subprocess.PIPE).stdout.decode('utf-8')
        
        # process output
        outputs = output.split(delimiter)
        if len(outputs) <= 2:
            print(output)
            # print("nn weights as follow")
            # print(nn_weights_list)
            raise NotImplementedError
        else:
            results_str = outputs[1].replace('\n', '').replace(
                'array', 'np.array')
            # print(collected_results_str)
            results = eval(results_str)

        gc.collect()
        return results
    
    def process_eval_result(self, curr_result_json):
        throughput = curr_result_json.get("throughput")
        obj = throughput
        return TrafficMAPFResult.from_raw(
            obj=obj, throughput=throughput
        )
    
    def actual_qd_score(self, objs):
        """Calculates QD score of the given objectives.

        Scores are normalized to be non-negative by subtracting a constant min
        score.

        Args:
            objs: List of objective values.
        """
        objs = np.array(objs)
        objs -= MIN_SCORE
        if np.any(objs < 0):
            warnings.warn("Some objective values are still negative.")
        return np.sum(objs)

if __name__ == "__main__":
    py_driver.run()