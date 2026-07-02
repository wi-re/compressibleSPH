
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


def computeCovariance(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    with record_function("[warpSPH] - computeCovariance"):
        C, Evals, L = computeRenormalizationMatrices(
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
        return C, Evals, L

def computeGradRhoL(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], L: Optional[RenormalizationState]) -> torch.Tensor:
    with record_function("[warpSPH] - (deltaSPH) - computeGradRhoL"):
        if L is None:
            C, Evals, L = computeCovariance(currentState, config, schemeConfig, adjacency)

        return warpOperation(
                currentState,
                OperationProperties(
                    kernel = config.kernel,
                    operation = WarpOperation.Gradient,
                    supportMode = SupportScheme.SuperSymmetric,
                    operationMode = OperationDirection.AllToAll,
                    gradientMode = GradientScheme.Difference
                ),
                queryValues = currentState.densities,
                # queryValues = testQuantity,
                domain = config.domain,
                adjacency = adjacency,
                renormalizationState = L
            )