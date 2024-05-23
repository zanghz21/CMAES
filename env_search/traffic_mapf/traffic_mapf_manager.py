import gin
import numpy as np
from dask.distributed import Client

import time
import logging
from logdir import LogDir

from env_search.utils.worker_state import init_traffic_mapf_module
from env_search.traffic_mapf.run import run_traffic_mapf, process_traffic_mapf_results
from env_search.traffic_mapf.module import TrafficMAPFModule
from env_search.traffic_mapf.config import TrafficMAPFConfig


logger = logging.getLogger(__name__)

@gin.configurable(denylist=["client", "rng"])
class TrafficMAPFManager:
    def __init__(self, 
                 client: Client, 
                 logdir: LogDir, 
                 rng: np.random.Generator=None, 
                 update_model_n_params: int = -1, 
                 bounds=None) -> None:
        
        self.iterative_update = True
        
        self.update_model_n_params = update_model_n_params
            
        self.logdir = logdir
        self.client = client
        self.rng = rng or np.random.default_rng()
        
        # Runtime
        self.repair_runtime = 0
        self.sim_runtime = 0
        
        self.module = TrafficMAPFModule(config := TrafficMAPFConfig())
        client.register_worker_callbacks(
            lambda: init_traffic_mapf_module(config))
        
    
    def get_sol_size(self):
        """Get number of parameters to optimize.
        """
        if self.iterative_update:
            return self.update_model_n_params
        else:
            raise NotImplementedError
        
    def eval_pipeline(self, unrepaired_sols, parent_sols=None, batch_idx=None):
        n_sols = len(unrepaired_sols)
        assert self.iterative_update
        
        iter_update_sols = unrepaired_sols
        evaluation_seeds = self.rng.integers(np.iinfo(np.int32).max / 2,
                                                size=len(iter_update_sols),
                                                endpoint=True)
        eval_logdir = self.logdir.pdir(
            f"evaluations/eval_batch_{batch_idx}")
        sim_start_time = time.time()
        sim_futures = [
            self.client.submit(
                run_traffic_mapf,
                nn_weights=sol, 
            ) for sol, seed in zip(iter_update_sols, evaluation_seeds)
        ]
        logger.info("Collecting evaluations")
        results = self.client.gather(sim_futures)
        self.sim_runtime += time.time() - sim_start_time
        
        results_json = []
        for i in range(n_sols):
            result_json = results[i]
            results_json.append(result_json)

        logger.info("Processing eval results")

        process_futures = [
            self.client.submit(
                process_traffic_mapf_results,
                curr_result_json=curr_result_json
            ) for curr_result_json in results_json
        ]
        results = self.client.gather(process_futures)
        return results
        
        