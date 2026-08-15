"""Public entry point for the delta-SPH density-diffusion term `drhodt_diss`:
scales `computeDensityDiffusionDeltaSPH`'s raw (unscaled) divergence output
by `delta * h / xi * c_s`, the standard delta-SPH prefactor, where `delta` is
`schemeConfig.diffusionParams.densityDelta`, `h` is the per-particle support,
`xi = sphKernel_xi(kernel, dim)` is the kernel's normalization constant, and
`c_s` is `schemeConfig.fluid.fixedSoundSpeed`. Which density-diffusion
formulation is evaluated (delta-SPH, non-renormalized, density-only, ...) is
selected via `schemeConfig.diffusionParams.densityDiffusionTerm`
(`DensityDiffusionScheme`) and forwarded to the underlying kernel; the caller
(`schemes/deltaSPH.py`) is responsible for only supplying `gradRho`/`gradRhoL`
when the selected scheme actually needs them.
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

from .wp_densityDelta import computeDensityDiffusionDeltaSPH

__all__ = ['computeDensityDiffusion']

def computeDensityDiffusion(currentState: Any, config: SimulationConfig, schemeConfig: Any, adjacency: Optional[Union[AdjacencyList, CompactHashMap]], gradRho: Optional[torch.Tensor], gradRhoL: Optional[torch.Tensor]) -> torch.Tensor:
    with record_function("[warpSPH] - (deltaSPH) - computeDensityDiffusion"):
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