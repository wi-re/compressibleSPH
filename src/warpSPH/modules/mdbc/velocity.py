
    # 

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from sphWarpCore import *

from sphWarpCore.kernels.wp_kernel import sphKernelDkDh, sphKernel_xi
from sphWarpCore.diffusion.viscosity import computePi_actual, DiffusionParameters

from warpSPH.configurations.moduleConfigurations.boundaryConditions import BCType
from warpSPH.configurations.region import RegionType
from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *
from ...configurations.moduleConfigurations.gravity import GravityType, gravityConfiguration
from ...configurations.weaklyCompressible import WeaklyCompressibleSPHConfig

from ..liu import interpolateLiuLiu


def noSlip(currentState: Any, config: SimulationConfig, schemeConfig: WeaklyCompressibleSPHConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    """
    Computes the no-slip boundary condition for ghost particles based on the velocities of fluid particles.
    """
    # Interpolate fluid velocities to ghost particles
    qVel = warpOperation(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Interpolate,
            supportMode = SupportScheme.Gather,
            operationMode = OperationDirection.FluidToGhost
        ),
        domain = config.domain,
        adjacency = adjacency,
        queryValues = currentState.velocities,
    )
    
    # Compute Shepard values for normalization
    shepValue = warpOperation(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Interpolate,
            supportMode = SupportScheme.Gather,
            operationMode = OperationDirection.FluidToGhost
        ),
        domain = config.domain,
        adjacency = adjacency,
        queryValues = torch.ones_like(currentState.densities),
    )

    # Normalize interpolated velocities
    qVel = qVel / (shepValue.view(-1,1) + 1e-7)

    # Compute the no-slip condition: u_g = 2 * u_f - u_g
    bodyVelocity = currentState.velocities
    bIndices = currentState.ghostIndices[currentState.kinds == 2]
    u_g = 2 * bodyVelocity - qVel

    r_ib = torch.linalg.norm(currentState.ghostOffsets, dim=-1)
    n_b = currentState.ghostOffsets / (r_ib.view(-1,1) + 1e-7)
    projected_vels = u_g - torch.einsum('nd, nd -> n', u_g, n_b).view(-1,1) * n_b

    u_g[bIndices,:] = projected_vels[currentState.kinds == 2,:]

    return u_g

def freeSlip(currentState: Any, config: SimulationConfig, schemeConfig: WeaklyCompressibleSPHConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    """
    Computes the free-slip boundary condition for ghost particles based on the velocities of fluid particles.
    """
    # Interpolate fluid velocities to ghost particles
    qVel = warpOperation(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Interpolate,
            supportMode = SupportScheme.Gather,
            operationMode = OperationDirection.FluidToGhost
        ),
        domain = config.domain,
        adjacency = adjacency,
        queryValues = currentState.velocities,
    )
    
    # Compute Shepard values for normalization
    shepValue = warpOperation(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Interpolate,
            supportMode = SupportScheme.Gather,
            operationMode = OperationDirection.FluidToGhost
        ),
        domain = config.domain,
        adjacency = adjacency,
        queryValues = torch.ones_like(currentState.densities),
    )

    # Normalize interpolated velocities
    qVel = qVel / (shepValue.view(-1,1) + 1e-7)

    # Compute the free-slip condition: u_g = u_f - 2 * (u_f . n_b) * n_b
    bodyVelocity = currentState.velocities
    bIndices = currentState.ghostIndices[currentState.kinds == 2]
    
    r_ib = torch.linalg.norm(currentState.ghostOffsets, dim=-1)
    n_b = currentState.ghostOffsets / (r_ib.view(-1,1) + 1e-7)

    projected_vels = qVel - torch.einsum('nd, nd -> n', qVel, n_b).view(-1,1) * n_b
    projected_vels[bIndices,:] = projected_vels[currentState.kinds == 2,:]


    return projected_vels

def extendedVelocity(currentState: Any, config: SimulationConfig, schemeConfig: WeaklyCompressibleSPHConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:

    uvs = [currentState.velocities[:,d] for d in range(currentState.velocities.shape[1])]

    extendedVelocities = [interpolateLiuLiu(
        currentState.positions[currentState.kinds == 2],
        referenceParticles = currentState,
        referenceQuantities = uv,
        config = config,
        neighbor_threshold = 4,
        direction = OperationDirection.FluidToGhost,
        supportScale = 1.0
    ) for uv in uvs] # res[:,0], res[:,1:], neighCounts, A_g, b


    ghostMask = currentState.kinds == 2
    relPos = currentState.ghostOffsets[ghostMask]

    velocities = [currentState.velocities.new_zeros(currentState.velocities.shape[0], device = currentState.velocities.device, dtype = currentState.velocities.dtype) for uv in extendedVelocities]

    for d in range(currentState.velocities.shape[1]):
        u_interp, u_interp_grad, numNeighbors, A_g, b = extendedVelocities[d]

        shepardNominator = b[:,0]
        shepardDenominator = A_g[:,0,0]
        shepardDensity = shepardNominator / shepardDenominator
        bIndices = currentState.ghostIndices[ghostMask]

        vel = velocities[d]
        vel[bIndices] = torch.where(numNeighbors > 0, shepardDensity, vel[bIndices])

        threshold = 9

        vel[bIndices] = torch.where(numNeighbors > threshold, (u_interp - torch.einsum('nu, nu -> n',(-relPos), u_interp_grad)), vel[ghostMask])

    extendedVelocities = torch.stack(velocities, dim = -1)
    return extendedVelocities


def constantVelocity(currentState: Any, config: SimulationConfig, schemeConfig: WeaklyCompressibleSPHConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    return currentState.velocities

def zeroVelocity(currentState: Any, config: SimulationConfig, schemeConfig: WeaklyCompressibleSPHConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    return currentState.velocities.new_zeros(currentState.velocities.shape[0], currentState.velocities.shape[1], device = currentState.velocities.device, dtype = currentState.velocities.dtype)


def computeBoundaryVelocities(currentState: Any, config: SimulationConfig, schemeConfig: WeaklyCompressibleSPHConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    if not torch.any(currentState.kinds == 1):
        return currentState.velocities
    with record_function("[warpSPH] - (mdbc) - computeBoundaryVelocities"):

        materials = currentState.materials[currentState.kinds == 1]
        uniqueMaterials = torch.unique(materials)
        
        
        boundaryRegions = [region for region in config.regions if region.type == RegionType.Boundary]
        kinds = [region.kind for region in boundaryRegions]
        ghostMask = currentState.kinds == 2
        boundaryMask = currentState.kinds == 1

        BCTypes = [region.kind for region in boundaryRegions]
        BCTypesT = torch.tensor([bct.value for bct in BCTypes], device = currentState.velocities.device, dtype = torch.int64)

        boundaryMaterials = currentState.materials.clone()
        boundaryMaterials[currentState.kinds != 1] = 0
        BCmask = BCTypesT[boundaryMaterials.long()]
        
        outputVelocities = currentState.velocities.clone()

        if BCType.zeros in BCTypes:
            zeroVelocities = zeroVelocity(currentState, config, schemeConfig, adjacency)
            mask = BCmask.view(-1,1) == BCType.zeros.value
            mask = mask & boundaryMask.view(-1,1)
            # print("Applying zero boundary condition to", torch.sum(mask).item(), "particles.")
            outputVelocities = torch.where(mask, zeroVelocities, outputVelocities)
        if BCType.constant in BCTypes:
            constantVelocities = constantVelocity(currentState, config, schemeConfig, adjacency)
            mask = BCmask.view(-1,1) == BCType.constant.value
            mask = mask & boundaryMask.view(-1,1)
            # print("Applying constant boundary condition to", torch.sum(mask).item(), "particles.")
            outputVelocities = torch.where(mask, constantVelocities, outputVelocities)
        if BCType.extended in BCTypes:
            extendedVelocities = extendedVelocity(currentState, config, schemeConfig, adjacency)
            mask = BCmask.view(-1,1) == BCType.extended.value
            mask = mask & boundaryMask.view(-1,1)
            # print("Applying extended boundary condition to", torch.sum(mask).item(), "particles.")
            outputVelocities = torch.where(mask, extendedVelocities, outputVelocities)        
        if BCType.noSlip in BCTypes:
            noSlipVelocities = noSlip(currentState, config, schemeConfig, adjacency)
            mask = BCmask.view(-1,1) == BCType.noSlip.value
            mask = mask & boundaryMask.view(-1,1)
            # print("Applying no-slip boundary condition to", torch.sum(mask).item(), "particles.")
            outputVelocities = torch.where(mask, noSlipVelocities, outputVelocities)            
        if BCType.freeSlip in BCTypes:
            freeSlipVelocities = freeSlip(currentState, config, schemeConfig, adjacency)
            mask = BCmask.view(-1,1) == BCType.freeSlip.value
            mask = mask & boundaryMask.view(-1,1)
            # print("Applying free-slip boundary condition to", torch.sum(mask).item(), "particles.")
            outputVelocities = torch.where(mask, freeSlipVelocities, outputVelocities)

        return outputVelocities
            


        # qVel = warpOperation(
        #     currentState,
        #     OperationProperties(
        #         kernel = config.kernel,
        #         operation = WarpOperation.Interpolate,
        #         supportMode = SupportScheme.Gather,
        #         operationMode = OperationDirection.FluidToGhost
        #     ),
        #     domain = config.domain,
        #     adjacency = adjacency,
        #     queryValues = currentState.velocities,
        # )
        # shepValue = warpOperation(
        #     currentState,
        #     OperationProperties(
        #         kernel = config.kernel,
        #         operation = WarpOperation.Interpolate,
        #         supportMode = SupportScheme.Gather,
        #         operationMode = OperationDirection.FluidToGhost
        #     ),
        #     domain = config.domain,
        #     adjacency = adjacency,
        #     queryValues = torch.ones_like(currentState.densities),
        # )

        # qVel = qVel / (shepValue.view(-1,1) + 1e-7)

        # bodyVelocity = currentState.velocities

        # bIndices = currentState.ghostIndices[currentState.kinds == 2]

        # u_g = 2 * bodyVelocity - qVel

        # r_ib = torch.linalg.norm(currentState.ghostOffsets, dim = -1)
        # n_b = currentState.ghostOffsets / (r_ib.view(-1,1) + 1e-7)

        # projected_vels = u_g - torch.einsum('nd, nd -> n', u_g, n_b).view(-1,1) * n_b

        # boundaryVelocities = currentState.velocities.clone()
        # boundaryVelocities[bIndices] = u_g[ghostMask]

        # projectedVelocities = currentState.velocities.clone()
        # projectedVelocities[bIndices] = projected_vels[ghostMask]

        # return boundaryVelocities, projectedVelocities