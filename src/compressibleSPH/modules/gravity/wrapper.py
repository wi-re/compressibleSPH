import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from sphWarpCore import *

from sphWarpCore.kernels.wp_kernel import sphKernelDkDh, sphKernel_xi
from sphWarpCore.diffusion.viscosity import computePi_actual, DiffusionParameters

from compressibleSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *
from ...configurations.moduleConfigurations.gravity import GravityType, gravityConfiguration

from .directional import computeDirectionalGravity
from .pointGravity import computePointGravity
from .potentialField import computePotentialFieldGravity

def computeGravity(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    if schemeConfig.gravityConfig.active is False:
        return torch.zeros_like(currentState.velocities)
    gravityType = schemeConfig.gravityConfig.gravityType
    if gravityType == GravityType.Directional:
        return computeDirectionalGravity(currentState, config, schemeConfig, adjacency)
    elif gravityType == GravityType.Point:
        return computePointGravity(currentState, config, schemeConfig, adjacency)
    elif gravityType == GravityType.PotentialField:
        return computePotentialFieldGravity(currentState, config, schemeConfig, adjacency)
    else:
        raise ValueError(f"Unsupported gravity type: {gravityType}")