

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *




from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *
from ...configurations.moduleConfigurations.surfaceDetection import SurfaceDetectionConfig

from .wp_dilate import dilateSurfaceMaskWarp


def computeLambdaGrad(
        currentState: Any, 
        
        
        config: SimulationConfig, schemeConfig: Any, surfaceConfig: SurfaceDetectionConfig, 
        
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]], 
        
        
        lambdas: Optional[torch.Tensor] = None, 
        renormalizationState: Optional[RenormalizationState] = None) -> torch.Tensor:
    if lambdas is None or renormalizationState is None:
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
        lambdas_ = torch.min(torch.abs(Evals), dim=-1).values
    else:
        lambdas_ = lambdas
        renormalizationState_ = renormalizationState

    return warpOperation(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Gradient,
            supportMode = SupportScheme.Scatter,
            operationMode = OperationDirection.AllToAll,
            gradientMode = GradientScheme.Difference
        ),
        queryValues = lambdas_,
        domain = config.domain,
        adjacency = adjacency,
        renormalizationState = renormalizationState_
    )

def computeNormalsLambdaGrad(
        currentState: Any, 
        
        
        config: SimulationConfig, schemeConfig: Any, surfaceConfig: SurfaceDetectionConfig, 
        
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]], 
        
        
        lambdas: Optional[torch.Tensor] = None, 
        renormalizationState: Optional[RenormalizationState] = None) -> torch.Tensor:
    lambdaGrad = computeLambdaGrad(
        currentState,
        config, schemeConfig, surfaceConfig,
        adjacency,
        lambdas = lambdas,
        renormalizationState = renormalizationState
    )
    return -torch.nn.functional.normalize(lambdaGrad, dim=-1)
