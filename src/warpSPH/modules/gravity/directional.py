import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from sphWarpCore import *




from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *
from ...configurations.moduleConfigurations.gravity import GravityType, gravityConfiguration


def computeDirectionalGravity(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    v = currentState.velocities
    direction = schemeConfig.gravityConfig.direction
    magnitude = schemeConfig.gravityConfig.magnitude
    if not isinstance(direction, torch.Tensor):
        direction = torch.tensor(direction, dtype = v.dtype, device = v.device)
    return (direction[:v.shape[1]] * magnitude).repeat(v.shape[0], 1)