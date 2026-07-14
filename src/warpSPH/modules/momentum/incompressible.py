from sphWarpCore import *
from ...systems.baseState import *
from warpSPH.configurations import SimulationConfig
from typing import Any, Optional, Union

from torch.profiler import profile, record_function, ProfilerActivity


def computeMomentumIncompressible(
        currentState: Any, 
        config: SimulationConfig, 
        schemeConfig: Any, 
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]], 
        advectionVelocities: torch.Tensor        
):
        # drhodt can be computed either using the current densities or the rest density
        # the latter is the option chosen in dfsph

    rho = schemeConfig.fluid.restDensity
#     rho = currentState.densities

    return - rho * warpOperation(
                currentState,
                OperationProperties(
                        kernel = config.kernel,
                        operation = WarpOperation.Divergence,
                        gradientMode = GradientScheme.Difference,
                        supportMode = SupportScheme.Scatter,
                ),
                queryValues = advectionVelocities,
                domain = config.domain,
                adjacency=adjacency,
                consistentDivergence = False,
        )       