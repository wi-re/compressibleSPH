"""The scalar wave equation on SPH particles.

Solves ``d2u/dt2 = c^2 laplacian(u)`` as a first-order system in ``(u, v)``,
with a PML-style absorbing term ``- damping * v`` folded into the acceleration
rather than applied after integration, which absorbs outgoing waves far better
than post-step damping does.

This is deliberately *not* a fluid scheme and has no registered case. It exists
as a demo of the SPH operators acting on unstructured, particle-like data --
a moving-neighbourhood Laplacian on scattered points -- which is the shape of
problem graph/point-cloud ML models are usually posed on. The heterogeneous
wave speed ``c`` and the randomisable sources/obstacles in
:mod:`warpSPH.configurations.waveEquationConfig` are there to make families of
such samples, not to model any particular physical setup.

See :mod:`warpSPH.systems.waveSystem` for the state it integrates and the
README section "The wave system" for how the pieces fit together.
"""

from ..systems import WaveSystemUpdatev3, WaveSystemv3
from warpSPHIntegrators import get_reference_state
from warpSPHCore import (
    GradientScheme, KernelFunctions, LaplacianScheme,
    OperationProperties, SupportScheme, WarpOperation,
    buildVerletList, warpOperation,
)

__all__ = ['f_wave_equation']


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

    # forcing = computeForcing(currentSystem, dt, config, compParams)
    # dvdt += forcing / currentState.masses.view(-1,1)


    update = WaveSystemUpdatev3(dudt=dudt, dvdt=dvdt)
    
    # enforceUpdates(update, state, dt, config, compParams)
    
    return update, adjacencyV
