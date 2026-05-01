from sphWarpCore import *
from ...systems.compressibleMonaghan import *
from compressibleSPH.config import SimulationConfig

from torch.profiler import profile, record_function, ProfilerActivity

def compute_dudt_warp(
    state: CompressibleState,
    config: SimulationConfig,
    supportScheme: SupportScheme = SupportScheme.Scatter,
    adjacency: Optional[AdjacencyList] = None,
    gradH: Optional[GradHState] = None,
):
    with record_function("warpSPH[compute_dudt]"):
        term = - state.pressures / state.densities
        if gradH is not None:
            term /= gradH.queryOmegas

        sphInterp = warpOperation(
            state,
            OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Divergence,
                gradientMode = GradientScheme.Difference,
                supportMode = supportScheme,
            ),
            queryValues = state.velocities,
            domain = config.domain,
            adjacency=adjacency,
            consistentDivergence = True,
        )
        return term * sphInterp
