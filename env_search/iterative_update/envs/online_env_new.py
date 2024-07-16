from env_search.competition.config import CompetitionConfig
from env_search.competition.update_model.utils import Map, comp_uncompress_vertex_matrix, comp_uncompress_edge_matrix
from env_search.utils import min_max_normalize, load_pibt_default_config, load_w_pibt_default_config, load_wppl_default_config, get_project_dir
from env_search.utils.logging import get_current_time_str, get_hash_file_name
from env_search.utils.task_generator import generate_task_and_agent
import numpy as np
import os
from gymnasium import spaces
import json
import time
import subprocess
import gc
from simulators.wppl.py_sim import py_sim
from env_search.iterative_update.envs.env import REDUNDANT_COMPETITION_KEYS
import shutil


DIRECTION2ID = {
    "R":0, "D":3, "L":2, "U":1, "W":4
}

class CompetitionOnlineEnvNew:
    def __init__(
        self,
        n_valid_vertices,
        n_valid_edges,
        config: CompetitionConfig,
        seed: int
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

    def update_paths_with_full_past(self, pos_hists, agents_paths):
        self.pos_hists = pos_hists
        self.move_hists = []
        for agent_path in agents_paths:
            self.move_hists.append(agent_path.replace(",", ""))
    
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
        for aid, (agent_path, agent_m) in enumerate(zip(results["exec_future"], results["exec_move"])):
            if aid in results["agents_finish_task"]:
                continue
            goal_id = results["final_tasks"][aid]
            for (x, y), m in zip(agent_path[1:], agent_m[1:]):
                if x*w+y == goal_id:
                    break
                d_id = DIRECTION2ID[m]
                exec_future_usage[d_id, x, y] += 1
        
        plan_future_usage = np.zeros((5, h, w))
        for aid, (agent_path, agent_m) in enumerate(zip(results["plan_future"], results["plan_move"])):
            if aid in results["agents_finish_task"]:
                continue
            goal_id = results["final_tasks"][aid]
            for (x, y), m in zip(agent_path, agent_m):
                if x*w+y == goal_id:
                    break
                d_id = DIRECTION2ID[m]
                plan_future_usage[d_id, x, y] += 1
                
        if exec_future_usage.sum()!=0:
            exec_future_usage = exec_future_usage/exec_future_usage.sum()
        if plan_future_usage.sum()!=0:
            plan_future_usage = plan_future_usage/plan_future_usage.sum()     
        
        return exec_future_usage, plan_future_usage
        
        
    def _gen_traffic_obs_new(self, is_init=False):
        h, w = self.comp_map.graph.shape
        edge_usage = np.zeros((4, h, w))
        wait_usage = np.zeros((1, h, w))
        
        if not is_init:
            time_range = min(self.config.past_traffic_interval, self.config.simulation_time-self.left_timesteps)
        else:
            time_range = min(self.config.past_traffic_interval, self.config.warmup_time)
        
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
            wait_usage = wait_usage/wait_usage.sum() * 100
        if edge_usage.sum() != 0:
            edge_usage = edge_usage/edge_usage.sum() * 100
        # print("new, wait_usage:", wait_usage.max(), "edge_usage:", edge_usage.max())
        return wait_usage, edge_usage


    def _gen_obs(self, result, is_init=False):
        wait_usage_matrix, edge_usage_matrix = self._gen_traffic_obs_new(is_init)
        # wait_usage_matrix, edge_usage_matrix = self._gen_traffic_obs(result)
        
        wait_costs = min_max_normalize(self.curr_wait_costs, 0.1, 1)
        edge_weights = min_max_normalize(self.curr_edge_weights, 0.1, 1)
        wait_cost_matrix = np.array(
            comp_uncompress_vertex_matrix(self.comp_map, wait_costs))
        edge_weight_matrix = np.array(
            comp_uncompress_edge_matrix(self.comp_map, edge_weights))
        # print(wait_costs.min(), wait_costs.max(), self.curr_wait_costs.min())
        # wait_cost_matrix = np.array(
        #     comp_uncompress_vertex_matrix(self.comp_map, self.curr_wait_costs))
        # edge_weight_matrix = np.array(
        #     comp_uncompress_edge_matrix(self.comp_map, self.curr_edge_weights))
        
        # wait_cost_matrix = min_max_normalize(wait_cost_matrix, 0.1, 1)
        # edge_weight_matrix = min_max_normalize(edge_weight_matrix, 0.1, 1)

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

            
    def get_base_kwargs(self):
        # TODO: SEED!!!
        kwargs = {
            "map_json_path": self.config.map_path,
            "simulation_steps": self.config.simulation_time,
            "gen_random": self.config.gen_random,
            "num_tasks": self.config.num_tasks,
            "num_agents": self.config.num_agents,
            "plan_time_limit": self.config.plan_time_limit,
            "seed": int(self.rng.integers(100000)),
            "task_dist_change_interval": self.config.task_dist_change_interval, 
            "preprocess_time_limit": self.config.preprocess_time_limit,
            "file_storage_path": self.config.file_storage_path,
            "task_assignment_strategy": self.config.task_assignment_strategy,
            "num_tasks_reveal": self.config.num_tasks_reveal, 
            "warmup_steps": self.config.warmup_time, 
            "update_gg_interval": self.config.update_interval
        }
        if not self.config.gen_random:
            file_dir = os.path.join(get_project_dir(), 'run_files', 'gen_task')
            os.makedirs(file_dir, exist_ok=True)
            sub_dir_name = get_hash_file_name()
            self.task_save_dir = os.path.join(file_dir, sub_dir_name)
            os.makedirs(self.task_save_dir, exist_ok=True)
            
            generate_task_and_agent(self.config.map_base_path, 
                total_task_num=100000, num_agents=self.config.num_agents, 
                save_dir=self.task_save_dir
            )
            
            kwargs["agents_path"] = os.path.join(self.task_save_dir, "test.agent")
            kwargs["tasks_path"] = os.path.join(self.task_save_dir, "test.task")
        if self.config.task_dist_change_interval > 0:
            kwargs["task_random_type"] = self.config.task_random_type
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
        return kwargs
    
        
    def _run_sim(self,
                 init_weight=False,
                 manually_clean_memory=True,
                 save_in_disk=True):
        """Run one simulation on the current edge weights and wait costs

        Args:
            init_weight (bool, optional): Whether the current simulation is on
                the initial weights. Defaults to False.

        """        
        # Initial weights are assumed to be valid
        if init_weight:
            edge_weights = self.curr_edge_weights.tolist()
            wait_costs = self.curr_wait_costs.tolist()
        else:
            edge_weights = min_max_normalize(self.curr_edge_weights, self.lb,
                                             self.ub).tolist()
            wait_costs = min_max_normalize(self.curr_wait_costs, self.lb,
                                           self.ub).tolist()
        
        kwargs = self.get_base_kwargs()
        kwargs["weights"] = json.dumps(edge_weights)
        kwargs["wait_costs"] = json.dumps(wait_costs)
        

        result_str = self.simulator.update_gg_and_step(edge_weights, wait_costs)
        result = json.loads(result_str)

        self.left_timesteps -= self.config.update_interval
        self.left_timesteps = max(0, self.left_timesteps)
        return result

    def step(self, action):
        self.i += 1  # increment timestep
        # print(f"[step={self.i}]")
        # The environment is fully observable, so the observation is the
        # current edge weights/wait costs
        wait_cost_update_vals = action[:self.n_valid_vertices]
        edge_weight_update_vals = action[self.n_valid_vertices:]
        self.curr_wait_costs = wait_cost_update_vals
        self.curr_edge_weights = edge_weight_update_vals

        result = self._run_sim()
        
        # self.last_agent_pos = result["final_pos"]
        # self.last_tasks = result["final_tasks"]
        assert self.starts is not None
        self.update_paths(result["actual_paths"])

        new_task_finished = result["num_task_finished"]
        reward = new_task_finished - self.num_task_finished
        self.num_task_finished = new_task_finished
        
        # terminated/truncate if no left time steps
        terminated = result["done"]
        truncated = terminated
        if terminated or truncated:
            if not self.config.gen_random:
                if os.path.exists(self.task_save_dir):
                    shutil.rmtree(self.task_save_dir)
                else:
                    raise NotImplementedError
        
        result["throughput"] = self.num_task_finished / self.config.simulation_time

        # Info includes the results
        sub_result = {k: v for k, v in result.items() if k not in REDUNDANT_COMPETITION_KEYS}
        info = {
            "result": sub_result,
            "curr_wait_costs": self.curr_wait_costs,
            "curr_edge_weights": self.curr_edge_weights,
        }

        return self._gen_obs(result), reward, terminated, truncated, info

    
    def reset(self, seed=None, options=None):
        self.i = 0
        self.num_task_finished = 0
        self.left_timesteps = self.config.simulation_time
        self.last_agent_pos = None
        # self.last_tasks = None
        
        self.starts = None
        self.task_save_dir = None
        
        self.pos_hists = [[] for _ in range(self.config.num_agents)]
        self.move_hists = [[] for _ in range(self.config.num_agents)]
        
        self.last_wait_usage = np.zeros(np.prod(self.comp_map.graph.shape))
        self.last_edge_usage = np.zeros(4*np.prod(self.comp_map.graph.shape))
        
        self.curr_edge_weights = np.ones(self.n_valid_edges)
        self.curr_wait_costs = np.ones(self.n_valid_vertices)
        
        kwargs = self.get_base_kwargs()
        kwargs["weights"] = json.dumps(self.curr_edge_weights.tolist())
        kwargs["wait_costs"] = json.dumps(self.curr_wait_costs.tolist())
        self.simulator = py_sim(**kwargs)
        result_str = self.simulator.warmup()
        result = json.loads(result_str)
        self.starts = result["starts"]
        self.update_paths(result["actual_paths"])

        obs = self._gen_obs(result, is_init=True)
        info = {"result": {}}
        return obs, info
    
if __name__ == "__main__":
    import gin
    from env_search.utils import get_n_valid_edges, get_n_valid_vertices
    from env_search.competition.update_model.utils import Map
    cfg_file_path = "config/competition/test_env.gin"
    gin.parse_config_file(cfg_file_path)
    cfg = CompetitionConfig()
    cfg.has_future_obs = False
    cfg.warmup_time = 300
    cfg.simulation_time = 1000
    cfg.update_interval = 20
    cfg.past_traffic_interval = 10
    cfg.task_dist_change_interval = 200
    
    # cfg.gen_random = False
    # cfg.map_base_path = "maps/competition/online_map/sortation_small.json"
    cfg.map_path = "maps/competition/online_map/sortation_small.json"
    
    
    comp_map = Map(cfg.map_path)
    domain = "competition"
    n_valid_vertices = get_n_valid_vertices(comp_map.graph, domain)
    n_valid_edges = get_n_valid_edges(comp_map.graph, bi_directed=True, domain=domain)
    
    env = CompetitionOnlineEnvNew(n_valid_vertices, n_valid_edges, cfg, seed=0)
    
    from env_search.competition.eval import vis_arr
    
    np.set_printoptions(threshold=np.inf)
    obs, info = env.reset()
    for i in range(4, 5):
        vis_arr(obs[i], name=f"step{env.i}_traffic{i}")
    
    done = False
    while not done:
        action = np.ones(n_valid_vertices+n_valid_edges)
        obs, reward, terminated, truncated, info = env.step(action)
        for i in range(4, 5):
            vis_arr(obs[i], name=f"step{env.i}_traffic{i}")
        done = terminated or truncated
    
    print(info["result"]["throughput"])
            