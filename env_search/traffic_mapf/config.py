import gin
from dataclasses import dataclass

@gin.configurable
@dataclass
class TrafficMAPFConfig:
    map_path: str = None
    all_json_path: str = None
    simu_time: int = 1000
    n_sim: int = 1
    
    # network cfg
    has_map: bool = False
    has_path: bool = False
    has_previous: bool = False
    use_all_flow: bool = True
    output_size: int = 4
    net_type: str = "quad"
    use_cached_nn: bool = False
    
    # sim cfg
    gen_tasks: bool = True
    num_agents: bool = gin.REQUIRED
    num_tasks: int = 100000
    seed: int = 0
    task_assignment_strategy: str = "roundrobin"
        