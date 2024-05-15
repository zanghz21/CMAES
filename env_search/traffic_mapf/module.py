import numpy as np
import json
import gc
import warnings
import subprocess

import time
from env_search.traffic_mapf.config import TrafficMAPFConfig
from env_search.traffic_mapf.result import TrafficMAPFResult
from env_search.utils import MIN_SCORE


class TrafficMAPFModule:
    def __init__(self, config: TrafficMAPFConfig):
        pass
        
    def evaluate(self, nn_weights: np.ndarray):
        nn_weights_list=nn_weights.tolist()
        # print(nn_weights_list)
        # raise NotImplementedError
        kwargs = {"network_params": json.dumps(nn_weights_list)}
        delimiter = "[=====delimiter======]"
        output = subprocess.run(
                    [
                        'python', '-c', f"""\
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
                    """
                    ],
                    stdout=subprocess.PIPE).stdout.decode('utf-8')
        
        outputs = output.split(delimiter)
        if len(outputs) <= 2:
            print(output)
            print("nn weights as follow")
            print(nn_weights_list)
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