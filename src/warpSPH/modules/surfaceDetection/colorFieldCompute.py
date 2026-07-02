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


# @torch.jit.script
# def computeColorField(
#     particles : WeaklyCompressibleState,
#     neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
#     supportScheme: SupportScheme = SupportScheme.Scatter
#     ):
#     with record_function("[SPH] - [Surface Detection] - Compute Color Field"):
#         ones = particles.positions.new_ones(particles.positions.shape[0])
#         term = SPHOperationCompiled(
#             particles,
#             quantity = (ones, ones),
#             neighborhood= neighborhood[0],
#             kernelValues = neighborhood[1],
#             operation= Operation.Interpolate,
#             supportScheme= supportScheme
#         )
#         termGrad = SPHOperationCompiled(
#             particles,
#             quantity = (term, term),
#             neighborhood= neighborhood[0],
#             kernelValues = neighborhood[1],
#             operation= Operation.Gradient,
#             gradientMode = GradientMode.Difference,
#             supportScheme= supportScheme
#         )
        
#         return term, termGrad


def computeColorField(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> Tuple[torch.Tensor, torch.Tensor]:
    ones = currentState.positions.new_ones(currentState.positions.shape[0])

    colorField = warpOperation(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Interpolate,
            supportMode = SupportScheme.Gather, # cullen switch E.1 in the CRK paper uses gather for density estimation
        ),
        queryValues = ones,
        domain = config.domain,
        adjacency = adjacency,
    )
    colorFieldGrad = warpOperation(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Gradient,
            supportMode = SupportScheme.Gather, # cullen switch E.1 in the CRK paper uses gather for density estimation
            gradientMode = GradientScheme.Difference,
        ),
        queryValues = colorField,
        domain = config.domain,
        adjacency = adjacency,
    )
    return colorField, colorFieldGrad
