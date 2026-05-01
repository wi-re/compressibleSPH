from sphWarpCore import *
from ...systems.baseState import *
from compressibleSPH.config import SimulationConfig

from torch.profiler import profile, record_function, ProfilerActivity

def computeMomentumConsistent_warp(
    state: BaseParticleState,
    config: SimulationConfig,
    supportScheme: SupportScheme = SupportScheme.Scatter,
    adjacency: Optional[AdjacencyList] = None,
    gradH: Optional[GradHState] = None,
):
    with record_function("warpSPH[computeMomentumConsistent]"):
        densities = state.densities 
        omega = gradH.queryOmegas if gradH is not None else None
        
        term = - densities / omega if gradH is not None else - densities

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