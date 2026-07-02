
import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from sphWarpCore import *

from sphWarpCore.kernels.wp_kernel import sphKernelDkDh, sphKernel_xi
from sphWarpCore.diffusion.viscosity import computePi_actual, DiffusionParameters

from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *
from ...configurations.moduleConfigurations.surfaceDetection import SurfaceDetectionConfig

from ..util.wp_sum import warpSum
from ..util.wp_numNeighbors import countNeighborsWarp

from ..surfaceDetection import *
from ..density import *
from .delta import computeDeltaShift
from ..util import *

from ...configurations.moduleConfigurations.shifting import ShiftProperties, ShiftingProjectionScheme



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

            with record_function(f"[warpSPH] - (shift) - computeDeltaShift"):
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
                        update[lMin < 0.4] = 0
                    else:
                        update[fsm > 0.5] = 0
                        update[lMin < 0.4] = 0
                        update[fs > 0.5] = 0
                
            # update = torch.clamp(update, -shiftingThreshold * spacing, shiftingThreshold * spacing)
            update[systemState.kinds != 0] = 0

            systemState.positions += update# * dt
                        
        dx = systemState.positions - initialPositions
        systemState.positions = initialPositions
        systemState.densities = initialDensities


        return dx
                