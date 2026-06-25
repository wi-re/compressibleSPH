
# @torch.jit.script
# def detectFreeSurfaceColorFieldGradient(
#     particles : WeaklyCompressibleState,
#     colorField : torch.Tensor, 
#     colorGrad: torch.Tensor,
#     xi :float,
#     colorFieldGradientThreshold : float,
#     neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
#     supportScheme: SupportScheme = SupportScheme.Scatter
#     ):
#     with record_function("[SPH] - [Surface Detection] - Detect Free Surface (Color Field Gradient)"):
        
#         fs = torch.linalg.norm(colorGrad, dim = -1) > colorFieldGradientThreshold * particles.supports / xi
#         return fs
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

from .maronneNormals import computeNormalsMaronne
from .wp_maronne import computeMaronneSurfaceDetection


from sphWarpCore.kernels.wp_kernel import sphKernelDkDh, sphKernel_xi
def detectFreeSurfaceMaronne(
        currentState: Any, 
        config: SimulationConfig, schemeConfig: Any, surfaceConfig: SurfaceDetectionConfig, 
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]], 

        renormalizationState: Optional[RenormalizationState] = None, 
        eigenValues: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None, 
        
        returnNormals: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if renormalizationState is None and normals is None:
        C, Evals, renormalizationState = computeRenormalizationMatrices(
            queryParticles = currentState,
            operationProperties = OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Gradient,
                operationMode = OperationDirection.AllToAll,
                supportMode = SupportScheme.SuperSymmetric
            ),
            domain = config.domain,
            adjacency = adjacency,
            returnEigVals = True
        )
    if normals is None:
        normals_maronne = computeNormalsMaronne(
            currentState,
            renormalizationState.renormalizationMatrices,
            config, schemeConfig, schemeConfig.surfaceDetectionConfig,
            adjacency,
        )
    else:
        normals_maronne = normals
    fsm = computeMaronneSurfaceDetection(
        currentState,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Gradient,
            operationMode = OperationDirection.AllToAll,
            supportMode = SupportScheme.SuperSymmetric
        ),
        domain = config.domain,
        adjacency = adjacency,
        surfaceNormals = normals_maronne,
    )

    return fsm, normals_maronne if returnNormals else fsm