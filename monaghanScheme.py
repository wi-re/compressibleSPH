from sphWarpCore import *
from compSystem import *
from compressibleSPH.config import SimulationConfig

def pressureForce_warp(
    state: CompressibleState,
    config: SimulationConfig,
    supportScheme: SupportScheme = SupportScheme.Scatter,
    adjacency: Optional[AdjacencyList] = None,
    gradH: Optional[GradHState] = None,
):
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

def compute_dudt_warp(
    state: CompressibleState,
    config: SimulationConfig,
    supportScheme: SupportScheme = SupportScheme.Scatter,
    adjacency: Optional[AdjacencyList] = None,
    gradH: Optional[GradHState] = None,
):
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

def computeMomentumConsistent_warp(
    state: CompressibleState,
    config: SimulationConfig,
    supportScheme: SupportScheme = SupportScheme.Scatter,
    adjacency: Optional[AdjacencyList] = None,
    gradH: Optional[GradHState] = None,
):
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