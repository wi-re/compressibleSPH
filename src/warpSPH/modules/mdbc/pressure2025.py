"""mDBC boundary-particle pressure extrapolation (`BoundaryPressureMode.
mdbcMlsPressure`).

Companion to `density2025.py`'s `computeMdbcDensity`, same Liu-Liu
(moving-least-squares) machinery, same ghost-point indirection (the MLS fit
is evaluated at each `kind==2` ghost point, then assigned back to its owning
`kind==1` boundary particle via `ghostIndices`), same low-neighbor-count
fallback ladder (Shepard 0th-order, then plain 0). Deliberately simpler than
`computeMdbcDensity`: there is no hydrostatic/gravity correction here, since
unlike density (which has a known rest value and an EOS relating it to
pressure) the pressure field has no reference value of its own to fall back
on -- it is *only* ever the incompressible pressure solve's own solution,
linearly extrapolated to the wall.

Called from `schemes/dfsph.py` right after `solveDivergenceFree`, once that
call has produced this step's fluid pressure field. The result is stored
back onto `currentState.pressures` for boundary particles, so it is available
to (a) `computePressureAccelIISPH`'s neighbor sums on the *next* step, giving
fluid particles near a wall a physically extrapolated pressure gradient
instead of an artificial zero-pressure boundary, and (b) as that next step's
warm start for the boundary rows the solver itself holds fixed at 0 during
the iteration (see `divergenceFree.py`'s `fluidMask` masking).
"""

import torch
from torch.profiler import record_function

from warpSPHCore import *

from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *

from ..liu import interpolateLiuLiu
from ._util import stateHasBoundaryParticles

__all__ = ['computeMdbcPressure']


def computeMdbcPressure(currentState, config: SimulationConfig, schemeConfig, adjacency):
    if not stateHasBoundaryParticles(currentState, config):
        return currentState.pressures
    with record_function("[warpSPH] - (mdbc) - computeMdbcPressure"):
        p_interp, p_interp_grad, numNeighbors, A_g, b = interpolateLiuLiu(
            currentState.positions[currentState.kinds == 2],
            referenceParticles=currentState,
            referenceQuantities=currentState.pressures,
            config=config,
            neighbor_threshold=4,
            direction=OperationDirection.FluidToGhost,
            supportScale=1.0,
            adjacency=adjacency.hashMap if isinstance(adjacency, AdjacencyList) else None,
        )

        ghostMask = currentState.kinds == 2
        bIndices = currentState.ghostIndices[ghostMask]

        # First-order Taylor correction from the ghost point (where the MLS
        # fit was evaluated) to the boundary particle's own position, same
        # `rho_interp + drho`-style projection `computeMdbcDensity` uses.
        relPos = -currentState.ghostOffsets[ghostMask]
        dP = -torch.einsum('nu, nu -> n', relPos, p_interp_grad)
        p_proj = p_interp + dP

        boundaryPressure = currentState.pressures.new_zeros(currentState.pressures.shape)
        shepardNominator = b[:, 0]
        shepardDenominator = A_g[:, 0, 0]
        shepardPressure = torch.where(
            shepardDenominator > 0, shepardNominator / shepardDenominator,
            torch.zeros_like(shepardDenominator))
        boundaryPressure[bIndices] = torch.where(
            numNeighbors > 1, shepardPressure, torch.zeros_like(shepardPressure))

        threshold = 9
        boundaryPressure[bIndices] = torch.where(
            numNeighbors > threshold, p_proj, boundaryPressure[bIndices])

        mergedPressures = currentState.pressures.clone()
        mergedPressures[bIndices] = boundaryPressure[bIndices]
        return mergedPressures
