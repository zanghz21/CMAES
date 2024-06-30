from env_search.competition.config import CompetitionConfig
from env_search.competition.update_model.utils import Map, comp_uncompress_vertex_matrix, comp_uncompress_edge_matrix
from env_search.utils import min_max_normalize, load_pibt_default_config, get_project_dir
from env_search.utils.logging import get_current_time_str
import numpy as np
import os
from gymnasium import spaces
import json
import hashlib
import time
import subprocess
import gc

class CompetitionOnlineEnv:
    def __init__(
        self,
        n_valid_vertices,
        n_valid_edges,
        config: CompetitionConfig,
        seed
    ):
        self.n_valid_vertices=n_valid_vertices
        self.n_valid_edges=n_valid_edges
        
        self.config = config
        assert(self.config.update_interval > 0)
        
        self.comp_map = Map(self.config.map_path)
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed

        # Use CNN observation
        h, w = self.comp_map.height, self.comp_map.width
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(10, h, w))

        if self.config.bounds is not None:
            self.lb, self.ub = self.config.bounds
        else:
            self.lb, self.ub = None, None

    def _gen_obs(self, result):
        edge_usage_matrix = np.array(result["edge_usage_matrix"])
        wait_usage_matrix = np.array(result["vertex_wait_matrix"])
        wait_cost_matrix = np.array(
            comp_uncompress_vertex_matrix(self.comp_map, self.curr_wait_costs))
        edge_weight_matrix = np.array(
            comp_uncompress_edge_matrix(self.comp_map, self.curr_edge_weights))

        wait_cost_matrix += self.last_wait_usage
        edge_usage_matrix += self.last_edge_usage
        
        self.last_wait_usage = wait_cost_matrix
        self.last_edge_usage = edge_usage_matrix
        
        # Normalize
        # wait_usage_matrix = min_max_normalize(wait_usage_matrix, 0, 1)
        # edge_usage_matrix = min_max_normalize(edge_usage_matrix, 0, 1)
        wait_usage_matrix = wait_cost_matrix/wait_usage_matrix.sum()
        edge_usage_matrix = edge_usage_matrix/edge_usage_matrix.sum()
        
        wait_cost_matrix = min_max_normalize(wait_cost_matrix, 0.1, 1)
        edge_weight_matrix = min_max_normalize(edge_weight_matrix, 0.1, 1)

        h, w = self.comp_map.height, self.comp_map.width
        edge_usage_matrix = edge_usage_matrix.reshape(h, w, 4)
        wait_usage_matrix = wait_usage_matrix.reshape(h, w, 1)
        edge_weight_matrix = edge_weight_matrix.reshape(h, w, 4)
        wait_cost_matrix = wait_cost_matrix.reshape(h, w, 1)
        obs = np.concatenate(
            [
                edge_usage_matrix,
                wait_usage_matrix,
                edge_weight_matrix,
                wait_cost_matrix,
            ],
            axis=2,
            dtype=np.float32,
        )
        obs = np.moveaxis(obs, 2, 0)
        return obs

    def _run_sim(self,
                 init_weight=False,
                 manually_clean_memory=True,
                 save_in_disk=True):
        """Run one simulation on the current edge weights and wait costs

        Args:
            init_weight (bool, optional): Whether the current simulation is on
                the initial weights. Defaults to False.

        """
        # cmd = f"./lifelong_comp --inputFile {self.input_file} --simulationTime {self.simulation_time} --planTimeLimit 1 --fileStoragePath large_files/"

        # Initial weights are assumed to be valid
        if init_weight:
            edge_weights = self.curr_edge_weights.tolist()
            wait_costs = self.curr_wait_costs.tolist()
        else:
            edge_weights = min_max_normalize(self.curr_edge_weights, self.lb,
                                             self.ub).tolist()
            wait_costs = min_max_normalize(self.curr_wait_costs, self.lb,
                                           self.ub).tolist()

        results = []
        
        simulation_steps = min(self.left_timesteps, self.config.update_interval)
        kwargs = {
            "map_json_path": self.config.map_path,
            "simulation_steps": simulation_steps,
            "gen_random": self.config.gen_random,
            "num_tasks": self.config.num_tasks,
            "num_agents": self.config.num_agents,
            "weights": json.dumps(edge_weights),
            "wait_costs": json.dumps(wait_costs),
            "plan_time_limit": self.config.plan_time_limit,
            # "seed": int(self.rng.integers(100000)),
            "preprocess_time_limit": self.config.preprocess_time_limit,
            "file_storage_path": self.config.file_storage_path,
            "task_assignment_strategy": self.config.task_assignment_strategy,
            "num_tasks_reveal": self.config.num_tasks_reveal,
            "config": load_pibt_default_config(),  # Use PIBT default config
        }
        
        if self.last_agent_pos is not None:
            kwargs["init_agent"] = True
            kwargs["init_agent_pos"] = str(self.last_agent_pos)
        
        if self.last_tasks is not None:
            kwargs["init_tasks"] = True
            kwargs["init_task_ids"] = str(self.last_tasks)


        if not manually_clean_memory:
            kwargs["seed"] = int(self.rng.integers(100000))
            result_jsonstr = wppl_py_driver.run(**kwargs)
            result = json.loads(result_jsonstr)
        else:
            if save_in_disk:
                file_dir = os.path.join(get_project_dir(), 'run_files')
                os.makedirs(file_dir, exist_ok=True)
                hash_obj = hashlib.sha256()
                raw_name = get_current_time_str().encode() + os.urandom(16)
                hash_obj.update(raw_name)
                file_name = hash_obj.hexdigest()
                file_path = os.path.join(file_dir, file_name)
                with open(file_path, 'w') as f:
                    json.dump(kwargs, f)
                t1 = time.time()
                delimiter1 = "----DELIMITER1----DELIMITER1----"
                delimiter2 = "----DELIMITER2----DELIMITER2----"
                output = subprocess.run(
                    [
                        'python', '-c', f"""\
import numpy as np
from simulators.wppl import py_driver as wppl_py_driver
import json
import time

file_path='{file_path}'
with open(file_path, 'r') as f:
    kwargs_ = json.load(f)

t0 = time.time()
rng = np.random.default_rng(seed={self.seed})
kwargs_["seed"] = int(rng.integers(100000))
t0 = time.time()
result_jsonstr = wppl_py_driver.run(**kwargs_)
t1 = time.time()
print("{delimiter2}")
print(t1-t0)
print("{delimiter2}")
result = json.loads(result_jsonstr)

np.set_printoptions(threshold=np.inf)

print("{delimiter1}")
print(result)
print("{delimiter1}")

                """
                    ],
                    stdout=subprocess.PIPE).stdout.decode('utf-8')
                t2 = time.time()
                if os.path.exists(file_path):
                    os.remove(file_path)
                else:
                    raise NotImplementedError

            else:
                t1 = time.time()
                delimiter1 = "----DELIMITER1----DELIMITER1----"
                output = subprocess.run(
                    [
                        'python', '-c', f"""\
import numpy as np
from simulators.wppl import py_driver as wppl_py_driver
import json

kwargs_ = {kwargs}
kwargs_["seed"] = int({self.rng.integers(100000)})
result_jsonstr = wppl_py_driver.run(**kwargs_)
result = json.loads(result_jsonstr)

np.set_printoptions(threshold=np.inf)
print("{delimiter1}")
print(result)
print("{delimiter1}")
                    """
                    ],
                    stdout=subprocess.PIPE).stdout.decode('utf-8')
                t2 = time.time()
            outputs = output.split(delimiter1)
            if len(outputs) <= 2:
                print(output)
                raise NotImplementedError
            else:
                result_str = outputs[1].replace('\n', '').replace(
                    'array', 'np.array')
                result = eval(result_str)

            gc.collect()
        self.left_timesteps -= simulation_steps
        return result

    def step(self, action):
        self.i += 1  # increment timestep

        # The environment is fully observable, so the observation is the
        # current edge weights/wait costs
        wait_cost_update_vals = action[:self.n_valid_vertices]
        edge_weight_update_vals = action[self.n_valid_vertices:]
        self.curr_wait_costs = wait_cost_update_vals
        self.curr_edge_weights = edge_weight_update_vals

        result = self._run_sim()
        self.num_task_finished += result["num_task_finished"]
        self.last_agent_pos = result["final_pos"]
        self.last_tasks = result["final_tasks"]

        # Reward is final step update throughput
        reward = 0
        
        # terminated/truncate if no left time steps
        terminated = (self.left_timesteps <= 0)
        truncated = terminated
        
        if terminated:
            reward = self.num_task_finished/self.config.simulation_time

        result["throughput"] = reward
        # Info includes the results
        info = {
            "result": result,
            "curr_wait_costs": self.curr_wait_costs,
            "curr_edge_weights": self.curr_edge_weights,
        }

        return self._gen_obs(result), reward, terminated, truncated, info

    
    def reset(self, seed=None, options=None):
        self.i = 0
        self.num_task_finished = 0
        self.left_timesteps = self.config.simulation_time
        self.last_agent_pos = None
        self.last_tasks = None
        
        zero_obs = np.zeros((10, *self.comp_map.graph.shape), dtype=np.float32)
        self.last_wait_usage = np.zeros(np.prod(self.comp_map.graph.shape))
        self.last_edge_usage = np.zeros(4*np.prod(self.comp_map.graph.shape))
        info = {"result": {}}
        return zero_obs, info
