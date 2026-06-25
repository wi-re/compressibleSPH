
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
from ...configurations.moduleConfigurations.surfaceDetection import SurfaceDetectionConfig

from .wp_dilate import dilateSurfaceMaskWarp


def dilateSurface(currentState: Any, freeSurfaceMask: torch.Tensor, config: SimulationConfig, schemeConfig: Any, surfaceConfig: SurfaceDetectionConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], overrideIterations: Optional[int] = None) -> torch.Tensor:

    out = freeSurfaceMask.clone()
    if overrideIterations is not None:
        iterations = overrideIterations
    else:
        iterations = surfaceConfig.expansionIterations
    for i in range(iterations):
        out = dilateSurfaceMaskWarp(
            currentState,
            OperationProperties(
                kernel = config.kernel,
                # operation = WarpOperation.DilateSurfaceMask,
                supportMode = SupportScheme.SuperSymmetric,
                operationMode = OperationDirection.AllToAll,
                gradientMode = GradientScheme.Naive
            ),
            freeSurfaceMask = out,
            domain = config.domain,
            adjacency = adjacency
        )
    return out