import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from sphWarpCore import *

from sphWarpCore.kernels.wp_kernel import sphKernelDkDh, sphKernel_xi
from sphWarpCore.diffusion.viscosity import computePi_actual, DiffusionParameters

from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *
from .wp_surfaceAware import computePressureSurfaceAwareWarp

def computePressureForceSurfaceAware(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    dvdt = - computePressureSurfaceAwareWarp(
        currentState,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = SupportScheme.SuperSymmetric,
        ),
        domain = config.domain,
        adjacency = adjacency,
        queryPressures = currentState.pressures,
        pressureTerm = schemeConfig.pressureForceTerm,
        querySurfaceMask = currentState.surfaceIndicators,
    ) / currentState.densities.view(-1,1)
    return dvdt