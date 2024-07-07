from env_search.competition.config import CompetitionConfig
from env_search.competition.update_model.utils import Map, comp_uncompress_vertex_matrix, comp_uncompress_edge_matrix
from env_search.utils import min_max_normalize, load_pibt_default_config, load_w_pibt_default_config, load_wppl_default_config, get_project_dir
from env_search.utils.logging import get_current_time_str
import numpy as np
import os
from gymnasium import spaces
import json
import hashlib
import time
import subprocess
import gc

DIRECTION2ID = {
    "R":0, "D":3, "L":2, "U":1, "W":4
}

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

    def update_paths(self, agents_paths):
        for agent_moves, agent_new_paths in zip(self.move_hists, agents_paths):
            for s in agent_new_paths:
                if s == ",":
                    continue
                agent_moves.append(s)
        
        for i, agent_pos in enumerate(self.pos_hists):
            if len(agent_pos) == 0:
                agent_pos.append(self.starts[i])
            
            last_h, last_w = agent_pos[-1]
            for s in agents_paths[i]:
                if s == ",":
                    continue
                elif s == "R":
                    cur_pos = [last_h, last_w+1]
                elif s == "D":
                    cur_pos = [last_h+1, last_w]
                elif s == "L":
                    cur_pos = [last_h, last_w-1]
                elif s == "U":
                    cur_pos = [last_h-1, last_w]
                elif s == "W":
                    cur_pos = [last_h, last_w]
                else:
                    print(f"s = {s}")
                    raise NotImplementedError
                assert (cur_pos[0]>=0 and cur_pos[0]<self.comp_map.height \
                    and cur_pos[1]>=0 and cur_pos[1]<self.comp_map.width)
                agent_pos.append(cur_pos)
                last_h, last_w = agent_pos[-1]     
        
    def _gen_future_obs(self, results):
        "exec_future, plan_future, exec_move, plan_move"
        # 5 dim
        h, w = self.comp_map.graph.shape
        exec_future_usage = np.zeros((5, h, w))
        for agent_path, agent_m in zip(results["exec_future"], results["exec_move"]):
            # print("exec:", agent_path, agent_m)
            for (x, y), m in zip(agent_path[1:], agent_m[1:]):
                d_id = DIRECTION2ID[m]
                exec_future_usage[d_id, x, y] += 1
        
        plan_future_usage = np.zeros((5, h, w))
        for agent_path, agent_m in zip(results["plan_future"], results["plan_move"]):
            # print("plan:", agent_path, agent_m)
            for (x, y), m in zip(agent_path, agent_m):
                d_id = DIRECTION2ID[m]
                plan_future_usage[d_id, x, y] += 1
                
        if exec_future_usage.sum()!=0:
            exec_future_usage = exec_future_usage/exec_future_usage.sum()
        if plan_future_usage.sum()!=0:
            plan_future_usage = plan_future_usage/plan_future_usage.sum()            
        
        return exec_future_usage, plan_future_usage
        
        
    def _gen_traffic_obs_new(self):
        h, w = self.comp_map.graph.shape
        edge_usage = np.zeros((4, h, w))
        wait_usage = np.zeros((1, h, w))
        
        time_range = min(self.config.past_traffic_interval, self.config.simulation_time-self.left_timesteps)
        for t in range(time_range):
            for agent_i in range(self.config.num_agents):
                prev_x, prev_y = self.pos_hists[agent_i][-(time_range+1-t)]
                # cur_x, cur_y = self.pos_hists[agent_i][-(self.config.past_traffic_interval-t)]
                
                move = self.move_hists[agent_i][-(time_range-t)]
                id = DIRECTION2ID[move]
                if id < 4:
                    edge_usage[id, prev_x, prev_y] += 1
                else:
                    wait_usage[0, prev_x, prev_y] += 1
        
        if wait_usage.sum() != 0:
            wait_usage_matrix = wait_usage/wait_usage.sum()
        if edge_usage.sum() != 0:
            edge_usage_matrix = edge_usage/edge_usage.sum()
        return wait_usage_matrix, edge_usage_matrix                       
                            
        
    def _gen_traffic_obs(self, result):
        edge_usage_matrix = np.array(result["edge_usage_matrix"])
        wait_usage_matrix = np.array(result["vertex_wait_matrix"])

        if self.config.use_cumulative_traffic:
            wait_usage_matrix += self.last_wait_usage
            edge_usage_matrix += self.last_edge_usage
        
        self.last_wait_usage = wait_usage_matrix
        self.last_edge_usage = edge_usage_matrix
        
        # Normalize
        # wait_usage_matrix = min_max_normalize(wait_usage_matrix, 0, 1)
        # edge_usage_matrix = min_max_normalize(edge_usage_matrix, 0, 1)
        
        wait_usage_matrix = wait_usage_matrix/wait_usage_matrix.sum()
        edge_usage_matrix = edge_usage_matrix/edge_usage_matrix.sum()
        
        h, w = self.comp_map.graph.shape
        edge_usage_matrix = edge_usage_matrix.reshape(h, w, 4)
        wait_usage_matrix = wait_usage_matrix.reshape(h, w, 1)
        
        edge_usage_matrix = np.moveaxis(edge_usage_matrix, 2, 0)
        wait_usage_matrix = np.moveaxis(wait_usage_matrix, 2, 0)
        return wait_usage_matrix, edge_usage_matrix
        
        
    def _gen_obs(self, result):
        wait_usage_matrix, edge_usage_matrix = self._gen_traffic_obs_new()
        # wait_usage_matrix, edge_usage_matrix = self._gen_traffic_obs(result)
        
        wait_cost_matrix = np.array(
            comp_uncompress_vertex_matrix(self.comp_map, self.curr_wait_costs))
        edge_weight_matrix = np.array(
            comp_uncompress_edge_matrix(self.comp_map, self.curr_edge_weights))
        
        wait_cost_matrix = min_max_normalize(wait_cost_matrix, 0.1, 1)
        edge_weight_matrix = min_max_normalize(edge_weight_matrix, 0.1, 1)

        h, w = self.comp_map.height, self.comp_map.width
        
        edge_weight_matrix = edge_weight_matrix.reshape(h, w, 4)
        wait_cost_matrix = wait_cost_matrix.reshape(h, w, 1)
        
        edge_weight_matrix = np.moveaxis(edge_weight_matrix, 2, 0)
        wait_cost_matrix = np.moveaxis(wait_cost_matrix, 2, 0)
        
        obs = np.concatenate(
            [
                edge_usage_matrix,
                wait_usage_matrix,
                edge_weight_matrix,
                wait_cost_matrix,
            ],
            axis=0,
            dtype=np.float32,
        )
        if self.config.has_future_obs:
            exec_future_usage, plan_future_usage = self._gen_future_obs(result)
            obs = np.concatenate([obs, exec_future_usage+plan_future_usage], axis=0, dtype=np.float32)
        # print("in step, obs.shape =", obs.shape)
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
            "num_tasks_reveal": self.config.num_tasks_reveal
        }
        if self.config.base_algo == "pibt":
            if self.config.has_future_obs:
                kwargs["config"] = load_w_pibt_default_config()
            else:
                kwargs["config"] = load_pibt_default_config()
        elif self.config.base_algo == "wppl":
            kwargs["config"] = load_wppl_default_config()
        else:
            print(f"base algo [{self.config.base_algo}] is not supported")
            raise NotImplementedError
        
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
        if self.starts is None:
            assert(self.i == 1)
            self.starts = result["starts"]
        self.update_paths(result["actual_paths"])

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
        self.pos_hists = [[] for _ in range(self.config.num_agents)]
        self.move_hists = [[] for _ in range(self.config.num_agents)]
        
        self.starts = None
        
        obs_dim = 10 if not self.config.has_future_obs else 15
        zero_obs = np.zeros((obs_dim, *self.comp_map.graph.shape), dtype=np.float32)
        # print("in reset, obs.shape =", zero_obs.shape)
        self.last_wait_usage = np.zeros(np.prod(self.comp_map.graph.shape))
        self.last_edge_usage = np.zeros(4*np.prod(self.comp_map.graph.shape))
        info = {"result": {}}
        return zero_obs, info


if __name__ == "__main__":
    import gin
    from env_search.utils import get_n_valid_edges, get_n_valid_vertices
    from env_search.competition.update_model.utils import Map
    cfg_file_path = "config/competition/test_env.gin"
    gin.parse_config_file(cfg_file_path)
    cfg = CompetitionConfig()
    cfg.has_future_obs = True
    comp_map = Map(cfg.map_path)
    domain = "competition"
    n_valid_vertices = get_n_valid_vertices(comp_map.graph, domain)
    n_valid_edges = get_n_valid_edges(comp_map.graph, bi_directed=True, domain=domain)
    
    env = CompetitionOnlineEnv(n_valid_vertices, n_valid_edges, cfg, seed=0)
    
    env.reset()
    
    done = False
    while not done:
        action = np.random.rand(n_valid_vertices+n_valid_edges)
        _, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
            