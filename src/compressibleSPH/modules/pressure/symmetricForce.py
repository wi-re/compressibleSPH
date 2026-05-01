from sphWarpCore import *
from ...systems.compressibleMonaghan import *
from compressibleSPH.config import SimulationConfig

from torch.profiler import profile, record_function, ProfilerActivity

def pressureForce_warp(
    state: CompressibleState,
    config: SimulationConfig,
    supportScheme: SupportScheme = SupportScheme.Scatter,
    adjacency: Optional[AdjacencyList] = None,
    gradH: Optional[GradHState] = None,
):
    with record_function("warpSPH[pressureForceSymmetric]"):
        return -warpOperation(
            state,
            OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Gradient,
                gradientMode = GradientScheme.Symmetric,
                supportMode = supportScheme,
            ),
            queryValues = state.pressures,
            domain = config.domain,
            adjacency=adjacency,
            gradHState = gradH
        ) / state.densities[:, None]
