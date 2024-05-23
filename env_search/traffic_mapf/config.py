import gin
from dataclasses import dataclass

@gin.configurable
@dataclass
class TrafficMAPFConfig:
    map_path: str = gin.REQUIRED
    simu_time: int = 1000
    
    # network cfg
    has_map: bool = False
    has_path: bool = False
    has_previous: bool = False
    use_all_flow: bool = True
    output_size: int = 4
    net_type: str = "quad"
        