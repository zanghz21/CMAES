import gin
from dataclasses import dataclass
from typing import Collection, Optional, Tuple, List, Callable, Dict


@gin.configurable
@dataclass
class TrafficMAPFConfig:
    map_path: str = None
    all_json_path: str = None
    simu_time: int = 1000
    n_sim: int = 1
    
    # network cfg
    rotate_input: bool = False
    win_r: int = 2
    has_map: bool = False
    has_path: bool = False
    has_previous: bool = False
    use_all_flow: bool = True
    output_size: int = 4
    hidden_size: int = 50
    net_type: str = "quad"
    use_cached_nn: bool = False
    default_obst_flow: float = 0.0
    learn_obst_flow: bool = False
    
    # sim cfg
    use_lns: bool = False
    gen_tasks: bool = True
    num_agents: bool = gin.REQUIRED
    num_tasks: int = 100000
    seed: int = 0
    task_assignment_strategy: str = "roundrobin"
    
    # offline sim cfg
    iter_update_n_iters: int = 2
    iter_update_mdl_kwargs: Dict = None
        