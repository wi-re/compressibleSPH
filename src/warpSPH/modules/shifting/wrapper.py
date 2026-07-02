
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
        adjacency = buildVerletList(
            systemState, 
            config.domain, verletScale = config.verletScale, supportMode = SupportScheme.SuperSymmetric,
            priorNeighborhood = adjacency,
            verbose = False)

        numNeighbors = countNeighbors(systemState, config, schemeConfig, adjacency)

        if summationDensity:
            systemState.densities = computeDensities(systemState, config, schemeConfig, adjacency)
            # ADD MDBC HERE
            
        if freeSurface:
            fs, fsm, n = detectFreeSurface(systemState, config, schemeConfig, schemeConfig.surfaceDetectionConfig, adjacency, returnNormals = True)

            C, Evals, renormalizationState_ = computeRenormalizationMatrices(
                queryParticles = systemState,
                operationProperties = OperationProperties(
                    kernel = config.kernel,
                    operation = WarpOperation.Gradient,
                    operationMode = OperationDirection.AllToAll,
                    supportMode = SupportScheme.SuperSymmetric
                ),
                domain = config.domain,
                adjacency = adjacency,
                returnEigVals = True
            )
            lMin = torch.min(torch.abs(Evals), dim = -1).values
        else:
            fs = fsm = n = lMin = None

        update, adjacency = computeDeltaShift(systemState, config, schemeConfig, domain, adjacency, iters = 1)
        # print(f"Iteration {i} [inside solveShifting], max shift magnitude: {update.norm(dim=1).max().item()}")


        if freeSurface:
            # lMin = lMin * float(eval_kernelScale(config.kernel.value, config.dim))
            if projectionScheme == ShiftingProjectionScheme.dot:
                result = update - torch.einsum('ij,ij->i', update, n).view(-1,1) * n
                update[fsm > 0.5] = result[fsm > 0.5] * surfaceScaling
                update[lMin < 0.4] = 0
            elif projectionScheme == ShiftingProjectionScheme.mat:
                nMat = torch.einsum('ij, ik -> ikj', n, n)
                M = torch.diag_embed(systemState.positions.new_ones(systemState.positions.shape)) - nMat
                result = torch.bmm(M, update.unsqueeze(-1)).squeeze(-1)
                
                # update[fsm > 0.5] = result[fsm > 0.5] * surfaceScaling * 5.0
                update[fsm > 0.5] = (lMin**2.0).view(-1,1)[fsm > 0.5] * result[fsm > 0.5]
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
    
    # adjacency = buildVerletList(
    #     systemState, 
    #     config.domain, verletScale = config.verletScale, supportMode = SupportScheme.SuperSymmetric,
    #     priorNeighborhood = adjacency,
    #     verbose = False)
    
    # du = dx / dt
    # rho = systemState.densities

    # drhodt_shift = warpOperation(
    #     systemState,
    #     operationProperties = OperationProperties(
    #         operation=WarpOperation.Divergence,
    #         kernel = config.kernel, 
    #         supportMode = SupportScheme.Gather,
    #         operationMode = OperationDirection.AllToAll,
    #         gradientMode = GradientScheme.Summation
    #     ),
    #     queryValues = rho.view(-1,1) * du,
    #     domain = domain,
    #     adjacency = adjacency
    # )

    # dudt = rho.view(-1,1) * warpOperation(
    #     systemState,
    #     operationProperties = OperationProperties(
    #         operation=WarpOperation.Divergence,
    #         kernel = config.kernel, 
    #         supportMode = SupportScheme.Gather,
    #         operationMode = OperationDirection.AllToAll,
    #         gradientMode = GradientScheme.Difference
    #     ),
    #     queryValues =  du,
    #     domain = domain,
    #     adjacency = adjacency
    # )

    # u3 = du.new_zeros((systemState.positions.shape[0], 3))
    # du3 = du.new_zeros((systemState.positions.shape[0], 3))
    # u3[:,:systemState.positions.shape[1]] = systemState.velocities
    # du3[:,:systemState.positions.shape[1]] = du
    # u_cross_du = torch.cross(u3, du3, dim = -1)
    # u_cross_du = u_cross_du[:,2]

    # duCross = rho.view(-1,1) * warpOperation(
    #     systemState,
    #     operationProperties = OperationProperties(
    #         operation=WarpOperation.Divergence,
    #         kernel = config.kernel, 
    #         supportMode = SupportScheme.Gather,
    #         operationMode = OperationDirection.AllToAll,
    #         gradientMode = GradientScheme.Summation
    #     ),
    #     queryValues =  u_cross_du,
    #     domain = domain,
    #     adjacency = adjacency
    # )


    return dx
            