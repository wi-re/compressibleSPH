
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
from ...configurations.surfaceDetection import SurfaceDetectionConfig

from .wp_sum import warpSum
from .wp_numNeighbors import countNeighborsWarp


# @torch.jit.script
def detectFreeSurfaceColorField(currentState: Any, colorField: torch.Tensor, numNeighbors: torch.Tensor, config: SimulationConfig, schemeConfig: Any, surfaceConfig: SurfaceDetectionConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    numNeighbors = countNeighborsWarp(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Interpolate,
            supportMode = SupportScheme.SuperSymmetric,
            operationMode = OperationDirection.AllToAll,
            gradientMode = GradientScheme.Naive
        ),
        domain = config.domain,
        adjacency = adjacency
    )
    meanColorField = warpSum(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Interpolate,
            supportMode = SupportScheme.SuperSymmetric,
            operationMode = OperationDirection.AllToAll,
            gradientMode = GradientScheme.Naive
        ),
        queryValues = colorField,
        domain = config.domain,
        adjacency = adjacency
    ) / numNeighbors
    # i, j = adjacency.i, adjacency.j
    # nj = currentState.numNeighbors
    # if nj is None:
    #     raise ValueError("nj is None. Please check the neighborhood.")
    # else:
    #     colorFieldMean = scatter_sum(colorField[j], i, dim = 0, dim_size = currentState.positions.shape[0]) / nj
    
    fs = torch.where((colorField < meanColorField) & (numNeighbors < config.targetNeighbors * surfaceConfig.colorFieldThreshold), 1., 0.)
    return fs