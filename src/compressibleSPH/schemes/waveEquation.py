from ..system import *
from ..utils import *
from integrators import *
from sphWarpCore import *


def f_wave_equation(system: WaveSystemv3, dt: float, verbose: bool = False):
    state = get_reference_state(system)

    # Compute neighborhood
    adjacencyV = buildVerletList(
        state,
        domain = system.domain, verletScale = 2**(1/state.positions.shape[1]),
        supportMode = SupportScheme.SuperSymmetric,
        priorNeighborhood = system.adjacency,
        verbose = verbose
    )

    laplacian_u = warpOperation(
        state, queryValues = state.u, 
        domain = system.domain, adjacency = adjacencyV, 
        operationProperties = OperationProperties(
            operation=WarpOperation.Laplacian,
            kernel = KernelFunctions.Wendland2, 
            supportMode = SupportScheme.SuperSymmetric,
            laplacianMode=LaplacianScheme.Brookshaw,
            gradientMode = GradientScheme.Difference
        ),
    )

    # Apply PML-style damping to the derivatives
    # This absorbs waves more effectively than post-integration damping
    dudt = state.v
    dvdt = state.c**2 * laplacian_u - state.damping * state.v

    return WaveSystemUpdatev3(dudt=dudt, dvdt=dvdt), adjacencyV
