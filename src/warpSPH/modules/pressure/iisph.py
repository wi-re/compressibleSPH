from sphWarpCore import *
from ...systems.compressibleMonaghan import *
from warpSPH.configurations import SimulationConfig

from torch.profiler import profile, record_function, ProfilerActivity

def computePressureAccelIISPH(
    state: CompressibleState,
    pressureValues: torch.Tensor,
    config: SimulationConfig,
    supportScheme: SupportScheme = SupportScheme.Scatter,
    adjacency: Optional[AdjacencyList] = None):
    return -warpOperation(
        state,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Gradient,
            gradientMode = GradientScheme.Summation,
            supportMode = supportScheme,
        ),
        queryValues = pressureValues,
        domain = config.domain,
        adjacency=adjacency,
    ) / state.densities[:, None]
