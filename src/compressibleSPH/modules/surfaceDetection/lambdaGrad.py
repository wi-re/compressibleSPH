

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

from .wp_dilate import dilateSurfaceMaskWarp


def computeLambdaGrad(currentState: Any, lambdas: torch.Tensor, config: SimulationConfig, schemeConfig: Any, surfaceConfig: SurfaceDetectionConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], renormalizationState: RenormalizationState) -> torch.Tensor:
    return warpOperation(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Gradient,
            supportMode = SupportScheme.Scatter,
            operationMode = OperationDirection.AllToAll,
            gradientMode = GradientScheme.Difference
        ),
        queryValues = lambdas,
        domain = config.domain,
        adjacency = adjacency,
        renormalizationState = renormalizationState
    )

# from diffSPH.enums import KernelCorrectionScheme
# @torch.jit.script
# def computeLambdaGrad(
#     particles : WeaklyCompressibleState,
#     L: torch.Tensor,
#     lambdas: torch.Tensor,
#     neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
#     supportScheme: SupportScheme = SupportScheme.Scatter
# ):
#     with record_function("[SPH] - [Surface Detection] - Compute Lambda Grad"):
#         return torch.nn.functional.normalize(SPHOperationCompiled(
#             particles,
#             quantity = (lambdas, lambdas),
#             neighborhood= neighborhood[0],
#             kernelValues = neighborhood[1],
#             operation= Operation.Gradient,
#             gradientMode = GradientMode.Difference,
#             supportScheme= supportScheme,
#             correctionTerms=[KernelCorrectionScheme.gradientRenorm]
#         ))
#         # return sph_op(particles, particles, domain, wrappedKernel, sparseNeighborhood, operation = 'gradient', gradientMode = 'difference', supportScheme = supportScheme, correctionTerms=[KernelCorrectionScheme.gradientRenorm], quantity = (lambdas, lambdas))
