"""Multi-iteration shifting driver: rebuilds the neighborhood, optionally
recomputes density and free-surface state, applies one `computeDeltaShift`
step per iteration, and (if `schemeConfig.surfaceDetectionConfig.active`)
projects the shift near the free surface before adding it to
`systemState.positions`.

Free-surface projection (`schemeConfig.shiftProperties.projectionScheme`,
`ShiftingProjectionScheme`) has three modes: `dot` removes the shift's normal
component and scales the tangential remainder by `surfaceScaling` for
surface particles; `mat` instead projects through a `(I - n n^T)` matrix and
scales by `lMin**2`; the fallback zeroes the shift outright for surface/
near-surface particles. All three additionally zero the shift wherever
`lMin < 0.4` (a fixed threshold, not currently exposed via config) and for
non-fluid particles (`kinds != 0`). Normals/`lMin` are recomputed from
`detectFreeSurface` each iteration unless `shiftProperties.reuseNormals` and
a prior surface state is already cached on `systemState`.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *




from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *
from ...configurations.moduleConfigurations.surfaceDetection import SurfaceDetectionConfig

from ..util.wp_sum import warpSum
from ..util.wp_numNeighbors import countNeighborsWarp

from ..surfaceDetection import *
from ..density import *
from .delta import computeDeltaShift
from .implicitShifting import computeImplicitShift, computeDynamicImplicitShift
from ..util import *

from ...configurations.moduleConfigurations.shifting import ShiftProperties, ShiftingProjectionScheme, ShiftingScheme

__all__ = ['solveShifting']


def solveShifting(
    systemState: Any,
    config: SimulationConfig, schemeConfig: Any,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]],
    dt: float,
    verbose: bool = False):
    with record_function("[warpSPH] - shift"):
        domain = config.domain
        kernel = config.kernel

        shiftIters = schemeConfig.shiftProperties.iterations
        summationDensity = schemeConfig.shiftProperties.summationDensity
        freeSurface = schemeConfig.surfaceDetectionConfig.active
        freeSurfaceScheme = schemeConfig.surfaceDetectionConfig.scheme
        normalScheme = schemeConfig.surfaceDetectionConfig.normalSource
        projectionScheme = schemeConfig.shiftProperties.projectionScheme
        surfaceScaling = schemeConfig.shiftProperties.surfaceScaling
        shiftingThreshold = schemeConfig.shiftProperties.threshold

        rho0 = schemeConfig.fluid.restDensity
        spacing = torch.pow(systemState.masses / rho0, 1/systemState.positions.shape[1]).mean().cpu().item()
        projectQuantities = schemeConfig.shiftProperties.projectQuantities

        initialPositions = systemState.positions.clone()
        initialDensities = systemState.densities.clone()

        for i in range(shiftIters):
            with record_function(f"[warpSPH] - (shift) - adjacency"):
                adjacency = buildVerletList(
                    systemState, 
                    config.domain, verletScale = config.verletScale, supportMode = SupportScheme.SuperSymmetric,
                    priorNeighborhood = adjacency,
                    verbose = False)

            with record_function(f"[warpSPH] - (shift) - countNeighbors"):
                numNeighbors = countNeighbors(systemState, config, schemeConfig, adjacency)

            if summationDensity:
                with record_function(f"[warpSPH] - (shift) - computeDensities"):
                    systemState.densities = computeDensities(systemState, config, schemeConfig, adjacency)
                # ADD MDBC HERE
                
            if freeSurface:
                with record_function(f"[warpSPH] - (shift) - detectFreeSurface"):
                    if schemeConfig.shiftProperties.reuseNormals and systemState.surfaceNormals is not None and systemState.surfaceLambdas is not None:
                        n = systemState.surfaceNormals
                        lMin = systemState.surfaceLambdas
                        surfaceIndicator = systemState.surfaceIndicators == 1
                    else:
                        fs, fsm, n, renormalizationState_, lMin = detectFreeSurface(systemState, config, schemeConfig, schemeConfig.surfaceDetectionConfig, adjacency, returnNormals = True)

                        surfaceIndicator = fsm > 0.5
                    # C, Evals, renormalizationState_ = computeRenormalizationMatrices(
                    #     queryParticles = systemState,
                    #     operationProperties = OperationProperties(
                    #         kernel = config.kernel,
                    #         operation = WarpOperation.Gradient,
                    #         operationMode = OperationDirection.AllToAll,
                    #         supportMode = SupportScheme.SuperSymmetric
                    #     ),
                    #     domain = config.domain,
                    #     adjacency = adjacency,
                    #     returnEigVals = True
                    # )
                    # lMin = torch.min(torch.abs(Evals), dim = -1).values
            else:
                fs = fsm = n = lMin = None

            with record_function(f"[warpSPH] - (shift) - computeShift"):
                if schemeConfig.shiftProperties.scheme == ShiftingScheme.implicit:
                    update, adjacency = computeImplicitShift(systemState, config, schemeConfig, domain, adjacency, iters = 1)
                elif schemeConfig.shiftProperties.scheme == ShiftingScheme.dynamic:
                    update, adjacency = computeDynamicImplicitShift(systemState, config, schemeConfig, domain, adjacency, iters = 1)
                else:
                    update, adjacency = computeDeltaShift(systemState, config, schemeConfig, domain, adjacency, iters = 1)
            # print(f"Iteration {i} [inside solveShifting], max shift magnitude: {update.norm(dim=1).max().item()}")


            if freeSurface:
                with record_function(f"[warpSPH] - (shift) - projectShift"):
                    # lMin = lMin * float(eval_kernelScale(config.kernel.value, config.dim))
                    if projectionScheme == ShiftingProjectionScheme.dot:
                        result = update - torch.einsum('ij,ij->i', update, n).view(-1,1) * n
                        update[fsm > 0.5] = result[fsm > 0.5] * surfaceScaling
                        update[lMin < 0.4] = 0
                    elif projectionScheme == ShiftingProjectionScheme.mat:
                        nMat = torch.einsum('ij, ik -> ikj', n, n)
                        M = torch.diag_embed(systemState.positions.new_ones(systemState.positions.shape)) - nMat
                        result = torch.bmm(M, update.unsqueeze(-1)).squeeze(-1)
                        
                        # update[surfaceIndicator] = result[surfaceIndicator] * surfaceScaling * 5.0
                        update[surfaceIndicator] = (lMin**2.0).view(-1,1)[surfaceIndicator] * result[surfaceIndicator]
                        # update[fs > 0.5] = result[fs> 0.5] * surfaceScaling
                        # update[surfaceIndicator] = 0.0
                        update[lMin < 0.4] = 0
                        update[surfaceIndicator]=0.0
                    else:
                        update[fsm > 0.5] = 0
                        update[lMin < 0.4] = 0
                        update[fs > 0.5] = 0
                
            update = torch.clamp(update, -shiftingThreshold * spacing, shiftingThreshold * spacing)
            update[systemState.kinds != 0] = 0

            systemState.positions += update# * dt
                        
        dx = systemState.positions - initialPositions
        systemState.positions = initialPositions
        systemState.densities = initialDensities


        return dx
                