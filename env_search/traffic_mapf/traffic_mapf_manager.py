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
                 n_evals: int = gin.REQUIRED, 
                 bounds=None) -> None:
        
        self.iterative_update = True
        self.n_evals = n_evals
        
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
        
        evaluation_seeds = self.rng.integers(np.iinfo(np.int32).max / 2,
                                                size=n_sols,
                                                endpoint=True)
        eval_logdir = self.logdir.pdir(
            f"evaluations/eval_batch_{batch_idx}")
    
        iter_update_sols = [sol for sol in unrepaired_sols for _ in range(self.n_evals)]
                    
        sim_start_time = time.time()
        sim_futures = [
            self.client.submit(
                run_traffic_mapf,
                nn_weights=sol, 
            ) for sol in iter_update_sols
        ]
        logger.info("Collecting evaluations")
        results = self.client.gather(sim_futures)
        self.sim_runtime += time.time() - sim_start_time
        
        # results_json = []
        # for i in range(n_sols):
        #     result_json = results[i]
        #     results_json.append(result_json)
        
        results_json_sorted = []
        for i in range(n_sols):
            curr_eval_results = []
            for j in range(self.n_evals):
                curr_eval_results.append(results[i * self.n_evals + j])
            results_json_sorted.append(curr_eval_results)

        logger.info("Processing eval results")

        process_futures = [
            self.client.submit(
                process_traffic_mapf_results,
                curr_result_jsons=curr_result_jsons
            ) for curr_result_jsons in results_json_sorted
        ]
        results = self.client.gather(process_futures)
        return results
        
        