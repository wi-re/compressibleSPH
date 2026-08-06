import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from sphWarpCore import *




from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *


def computeGradRho(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    with record_function("[warpSPH] - computeGradRho"):
        return warpOperation(
                currentState,
                OperationProperties(
                    kernel = config.kernel,
                    operation = WarpOperation.Gradient,
                    supportMode = SupportScheme.SuperSymmetric,
                    operationMode = OperationDirection.AllToAll,
                    gradientMode = GradientScheme.Naive
                ),
                queryValues = currentState.densities,
                # queryValues = testQuantity,
                domain = config.domain,
                adjacency = adjacency,
                # renormalizationState = RenormalizationState(C=C, eigVals=Evals, L=L)
            )