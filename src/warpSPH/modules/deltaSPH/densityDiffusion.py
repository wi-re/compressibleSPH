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

from .wp_densityDelta import computeDensityDiffusionDeltaSPH

def computeDensityDiffusion(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], gradRho: Optional[torch.Tensor], gradRhoL: Optional[torch.Tensor]) -> torch.Tensor:
    delta = schemeConfig.diffusionParams.densityDelta
    xi = sphKernel_xi(config.kernel.value, config.dim)
    drhodt_scaling = delta * currentState.supports / xi * schemeConfig.fluid.fixedSoundSpeed
    drhodt_diss = drhodt_scaling * computeDensityDiffusionDeltaSPH(
        currentState,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Divergence,
            supportMode = SupportScheme.SuperSymmetric,
            operationMode = OperationDirection.AllToAll,
        ),
        domain = config.domain,
        adjacency = adjacency,
        queryGradRho = gradRho,
        queryGradRhoL = gradRhoL,
        densityScheme = schemeConfig.diffusionParams.densityDiffusionTerm
    )
    return drhodt_diss