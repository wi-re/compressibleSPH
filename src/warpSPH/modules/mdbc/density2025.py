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

from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *
from ...configurations.moduleConfigurations.gravity import GravityType, gravityConfiguration

from ..liu import interpolateLiuLiu

def computeMdbcDensity(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    if not torch.any(currentState.kinds == 1):
        return currentState.densities
    with record_function("[warpSPH] - (mdbc) - computeMdbcDensity"):
        rho_interp, rho_interp_grad, numNeighbors, A_g, b = interpolateLiuLiu(
            currentState.positions[currentState.kinds == 2],
            referenceParticles = currentState,
            referenceQuantities = currentState.densities,
            config = config,
            neighbor_threshold = 4,
            direction = OperationDirection.FluidToGhost,
            supportScale = 1.0,
            adjacency = adjacency.hashMap if isinstance(adjacency, AdjacencyList) else None
        )
        # return res[:,0], res[:,1:], neighCounts

        ghostMask = currentState.kinds == 2
        bIndices = currentState.ghostIndices[ghostMask]

        rho0 = schemeConfig.fluid.restDensity
        c_s = schemeConfig.fluid.fixedSoundSpeed
        g = torch.tensor(schemeConfig.gravityConfig.direction, dtype = currentState.positions.dtype, device = currentState.positions.device) * schemeConfig.gravityConfig.magnitude if schemeConfig.gravityConfig.active else torch.zeros_like(currentState.positions[0])


        boundaryDensity = currentState.densities.new_ones(currentState.densities.shape) * schemeConfig.fluid.restDensity
        shepardNominator = b[:,0]
        shepardDenominator = A_g[:,0,0]
        shepardDensity = torch.where(shepardDenominator > 0, shepardNominator / shepardDenominator, rho0)

        boundaryDensity[bIndices] = torch.where(numNeighbors > 1, shepardDensity, rho0)

        rho_g = boundaryDensity[bIndices]
        P_g = c_s**2 * (rho_g - rho0)


        relPos = -currentState.ghostOffsets[ghostMask]
        nb = relPos
        # This normalization is not in the paper https://www.sciencedirect.com/science/article/pii/S0045793025003305?via%3Dihub
        # But it is correct, see the dualsphysics code and thanks Aaron!
        nb = torch.nn.functional.normalize(nb, dim = 1)

        dot = torch.einsum('ni, i -> n', nb, g)
        dot2 = torch.einsum('ni, ni -> n', relPos, nb)

        # if the normal is pointing in the direction of the gravity then we disable the gravity contribution as we want to avoid negative densities in the boundary particles
        # dot = torch.where(dot > 0, torch.zeros_like(dot), dot)

        P_b = P_g + rho0 * (dot * dot2)
        rho_b = rho0 + P_b / c_s**2
        rho_b = torch.clamp(rho_b, min = rho0)

        # rho_b = torch.where(numNeighbors > 4, rho_b, boundaryDensity[bIndices])

        # boundaryDensity[bIndices] = torch.where(numNeighbors > 4, rho_b, boundaryDensity[bIndices])
        boundaryDensity[bIndices] = rho_b
        threshold = 9

        drho = -torch.einsum('nu, nu -> n',(relPos), rho_interp_grad)
        rho_proj = (rho_interp + drho)
        # rho_proj = torch.where(drho < rho0 * (dot * dot2), rho_b, rho_proj)
        boundaryDensity[bIndices] = torch.where(numNeighbors > threshold, rho_proj, boundaryDensity[bIndices])

        mergedDensitities = currentState.densities.clone()
        mergedDensitities[bIndices] = boundaryDensity[bIndices]

        # print(f'Mdbc densities: min={boundaryDensity[bIndices].min():.3g}, max={boundaryDensity[bIndices].max():.3g}, mean={boundaryDensity[bIndices].mean():.3g}')
        return mergedDensitities