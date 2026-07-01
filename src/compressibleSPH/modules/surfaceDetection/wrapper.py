from .colorFieldDetection import detectFreeSurfaceColorField
from .colorFieldGradientDetection import detectFreeSurfaceColorFieldGradient
from .colorFieldCompute import computeColorField

from .dilation import dilateSurface
from .lambdaGrad import computeLambdaGrad, computeNormalsLambdaGrad

from .maronneNormals import computeNormalsMaronne

from .barecascoDetection import detectFreeSurfaceBarecasco
from .maronneDetection import detectFreeSurfaceMaronne


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
from ...configurations.moduleConfigurations.surfaceDetection import SurfaceDetectionConfig, SurfaceDetectionScheme, NormalSource


def computeNormals(
    currentState: Any,
    config: SimulationConfig, schemeConfig: Any, surfaceConfig: SurfaceDetectionConfig,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]],
    renormalizationState: Optional[RenormalizationState] = None
) -> torch.Tensor:
    if surfaceConfig.normalSource == NormalSource.LambdaGrad or surfaceConfig.normalSource == NormalSource.Maronne: 
        C, Evals, renormalizationState_ = computeRenormalizationMatrices(
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
    else:
        renormalizationState_ = renormalizationState

    if surfaceConfig.normalSource == NormalSource.ColorFieldGrad:
        colorField, colorFieldGrad = computeColorField(
            currentState,
            config, schemeConfig,
            adjacency,
        )
        normals = -torch.nn.functional.normalize(colorFieldGrad, dim = 1)
    elif surfaceConfig.normalSource == NormalSource.LambdaGrad:
        normals = computeNormalsLambdaGrad(
            currentState,
            config, schemeConfig, surfaceConfig,
            adjacency,
            renormalizationState = renormalizationState_
        )
    elif surfaceConfig.normalSource == NormalSource.Maronne:
        normals = computeNormalsMaronne(
            currentState,
            renormalizationState_.renormalizationMatrices,
            config, schemeConfig, surfaceConfig,
            adjacency,
        )
    elif surfaceConfig.normalSource == NormalSource.Native:
        normals = None
    else:
        raise ValueError(f"Unknown normal source: {surfaceConfig.normalSource}")
    return normals, renormalizationState_
    

def detectFreeSurface(
    currentState: Any,
    config: SimulationConfig, schemeConfig: Any, surfaceConfig: SurfaceDetectionConfig,

    adjacency: Optional[Union[AdjacencyList, CompactHashMap]],
    renormalizationState: Optional[RenormalizationState] = None,
    returnNormals: bool = True
):
    if surfaceConfig.active == False:
        fsm = torch.zeros(currentState.positions.shape[0], device = currentState.positions.device, dtype = currentState.positions.dtype)
        normals = torch.zeros_like(currentState.positions)
        return (fsm, fsm) if not returnNormals else (fsm, fsm, normals)

    normals, renormalizationState_ = computeNormals(
        currentState,
        config, schemeConfig, surfaceConfig,
        adjacency,
        renormalizationState = renormalizationState
    )

    if surfaceConfig.scheme == SurfaceDetectionScheme.ColorField:
        fsm, normals2 = detectFreeSurfaceColorField(
            currentState,
            config, schemeConfig, surfaceConfig,
            adjacency,
            returnNormals = True
        )
    elif surfaceConfig.scheme == SurfaceDetectionScheme.ColorFieldGrad:
        fsm, normals2 = detectFreeSurfaceColorFieldGradient(
            currentState,
            config, schemeConfig, surfaceConfig,
            adjacency,
            returnNormals = True
        )
    elif surfaceConfig.scheme == SurfaceDetectionScheme.Barecasco:
        fsm, normals2 = detectFreeSurfaceBarecasco(
            currentState,
            config, schemeConfig, surfaceConfig,
            adjacency,
            returnNormals = True
        )
    elif surfaceConfig.scheme == SurfaceDetectionScheme.Maronne:
        fsm, normals2 = detectFreeSurfaceMaronne(
            currentState,
            config, schemeConfig, surfaceConfig,
            adjacency,
            renormalizationState = renormalizationState_,
            normals = normals,
            returnNormals = True
        )
    else:
        raise ValueError(f"Unknown surface detection scheme: {surfaceConfig.scheme}")

    if surfaceConfig.normalSource == NormalSource.Native:
        normals = normals2

    fs = fsm.clone().to(dtype = currentState.positions.dtype, device = currentState.positions.device)
    for i in range(surfaceConfig.expansionIterations):
        fs = dilateSurface(
            currentState, fs,
            config, schemeConfig, surfaceConfig,
            adjacency,
            overrideIterations = 1
        )

    return (fsm, fs) if not returnNormals else (fsm, fs, normals)