

import torch

from ...systems.compressibleMonaghan import CompressibleSystemUpdate
from ...configurations.compSPHConfig import CompSPHConfig
from ...configurations.simulationConfig import SimulationConfig
from ...enumTypes import EnergyScheme
from sphWarpCore import *

from typing import Any, List, Optional
import numpy as np
# ------ Beginning of scatter functionality ------ #
# Scatter summation functionality based on pytorch geometric scatter functionality
# This is included here to make the code independent of pytorch geometric for portability
# Note that pytorch geometric is licensed under an MIT licenses for the PyG Team <team@pyg.org>
@torch.jit.script
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

from torch.profiler import record_function

@torch.jit.script
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    with record_function("scatter_sum"):
        index = broadcast(index, src, dim)
        if out is None:
            size = list(src.size())
            if dim_size is not None:
                size[dim] = dim_size
            elif index.numel() == 0:
                size[dim] = 0
            else:
                size[dim] = int(index.max()) + 1
            out = torch.zeros(size, dtype=src.dtype, device=src.device)
            return out.scatter_add_(dim, index, src)
        else:
            return out.scatter_add_(dim, index, src)
        

from typing import Tuple
def compSPH_deltaU_multistep(
        dt : float,
        initialState: Any, #CompSPHState,
        returnValues: Any, # List[Tuple[AdjacencyList, CompSPHState]],
        updates : List[CompressibleSystemUpdate],
        butcherTerms: np.ndarray,
        generalConfig: SimulationConfig,
        solverConfig: CompSPHConfig,
        verbose: bool = False
):
    # verbosePrint(verbose, '[DeltaU] Computing Accelerations')
    fullAcceleration = torch.zeros_like(initialState.velocities)
    for i, update in enumerate(updates):
        # print(f'[DeltaU]\tStep {i:2d}/{len(updates):2d}\tk = {butcherTerms[i]:.2f}')
        # print(f'dudt shape: {update.dudt.shape}')
        fullAcceleration += butcherTerms[i] * update.dvdt
    halfStepVelocity = initialState.velocities + 0.5 * fullAcceleration * dt

    deltaU = torch.zeros_like(initialState.internalEnergies)
    # verbosePrint(verbose, '[DeltaU] Computing DeltaU')
    for ii, (stepAdjacency, stepState) in enumerate(returnValues):
        # verbosePrint(verbose, f'[DeltaU]\tStep {ii:2d}/{len(particleStates):2d}\tk = {butcherTerms[ii]:.2f}')
        i = stepAdjacency.i
        j = stepAdjacency.j

        v_i = halfStepVelocity[i]
        v_j = halfStepVelocity[j]
        v_ji = v_j - v_i

        f_ij = stepState.f_ij
        ap_ij = stepState.ap_ij
        av_ij = stepState.av_ij
        
        k = butcherTerms[ii]
        if solverConfig.energyScheme == EnergyScheme.PdV:
            term = k * f_ij * torch.einsum('ij, ij -> i', v_ji, ap_ij)
            term += k * 1/2 * torch.einsum('ij, ij -> i', v_ji, av_ij)
        else:
            term = k * f_ij * torch.einsum('ij, ij -> i', v_ji, ap_ij + av_ij)
            
        deltaU += scatter_sum(term, i, dim = 0, dim_size = initialState.positions.shape[0])
    # verbosePrint(verbose, '[DeltaU] Done')
    return deltaU

