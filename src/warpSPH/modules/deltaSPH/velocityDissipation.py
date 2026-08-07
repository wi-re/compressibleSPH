import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *




from warpSPH.configurations.simulationConfig import SimulationConfig
from warpSPH.modules.deltaSPH.wp_viscosityDelta import computeVelocityDiffusionDeltaSPH
from ...enumTypes import *

from .wp_densityDelta import computeDensityDiffusionDeltaSPH

def computeVelocityDiffusion(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    with record_function("[warpSPH] - (deltaSPH) - computeVelocityDiffusion"):
        dvdt_diss = computeVelocityDiffusionDeltaSPH(
            currentState,
            operationProperties = OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Laplacian,
                supportMode = SupportScheme.SuperSymmetric,
                operationMode = OperationDirection.AllToAll,
            ),
            domain = config.domain,
            adjacency = adjacency,
            queryVelocities = currentState.velocities,
            inviscid = schemeConfig.diffusionParams.inviscid,
            c_s = schemeConfig.fluid.fixedSoundSpeed,
            alpha = schemeConfig.diffusionParams.inviscidAlpha,
            nu = schemeConfig.diffusionParams.viscidNu,
        )
        return dvdt_diss