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


def computeMomentum(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    return -currentState.densities * warpOperation(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Divergence,
            supportMode = SupportScheme.SuperSymmetric,
            operationMode = OperationDirection.AllToAll,
            gradientMode = GradientScheme.Difference
        ),
        queryValues = currentState.velocities,
        domain = config.domain,
        adjacency = adjacency,
        consistentDivergence = False
    )