import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *




from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *
from ...configurations.moduleConfigurations.gravity import GravityType, gravityConfiguration

from .directional import computeDirectionalGravity
from .pointGravity import computePointGravity
from .potentialField import computePotentialFieldGravity

def computeGravity(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    with record_function("[warpSPH] - computeGravity"):
        if schemeConfig.gravityConfig.active is False:
            return torch.zeros_like(currentState.velocities)
        gravityType = schemeConfig.gravityConfig.type
        if gravityType == GravityType.Directional:
            return computeDirectionalGravity(currentState, config, schemeConfig, adjacency)
        elif gravityType == GravityType.PointSource:
            return computePointGravity(currentState, config, schemeConfig, adjacency)
        elif gravityType == GravityType.PotentialField:
            return computePotentialFieldGravity(currentState, config, schemeConfig, adjacency)
        else:
            raise ValueError(f"Unsupported gravity type: {gravityType}")