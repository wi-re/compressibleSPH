from warpSPHCore import *

from warpSPH.systems.compressibleMonaghan import CompressibleState
from ...systems.baseState import *
from warpSPH.configurations import SimulationConfig
from typing import Any, Optional, Union


from torch.profiler import profile, record_function, ProfilerActivity




def computePressureShiftIISPH(
    state: CompressibleState,
    config: SimulationConfig,
    pressureAccels: torch.Tensor,
    supportScheme: SupportScheme = SupportScheme.Scatter,
    adjacency: Optional[AdjacencyList] = None,):
    kernelSum = warpOperation(
        state,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Divergence,
            gradientMode = GradientScheme.Difference,
            supportMode = SupportScheme.Scatter,
        ),
        queryValues = pressureAccels,
        domain = config.domain,
        adjacency=adjacency,
        consistentDivergence = False, # Computes fj - fi, (9) in PBSPH states fi -fj thus the minus sign
    ) #* state.densities
    return - kernelSum
