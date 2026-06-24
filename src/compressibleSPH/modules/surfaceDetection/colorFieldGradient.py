
# @torch.jit.script
# def detectFreeSurfaceColorFieldGradient(
#     particles : WeaklyCompressibleState,
#     colorField : torch.Tensor, 
#     colorGrad: torch.Tensor,
#     xi :float,
#     colorFieldGradientThreshold : float,
#     neighborhood: Tuple[SparseNeighborhood, PrecomputedNeighborhood],
#     supportScheme: SupportScheme = SupportScheme.Scatter
#     ):
#     with record_function("[SPH] - [Surface Detection] - Detect Free Surface (Color Field Gradient)"):
        
#         fs = torch.linalg.norm(colorGrad, dim = -1) > colorFieldGradientThreshold * particles.supports / xi
#         return fs
import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from sphWarpCore import *

from sphWarpCore.kernels.wp_kernel import sphKernelDkDh, sphKernel_xi
from sphWarpCore.diffusion.viscosity import computePi_actual, DiffusionParameters

from compressibleSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *
from ...configurations.surfaceDetection import SurfaceDetectionConfig



from sphWarpCore.kernels.wp_kernel import sphKernelDkDh, sphKernel_xi
def detectFreeSurfaceColorFieldGradient(currentState: Any, colorField: torch.Tensor, colorGrad: torch.Tensor, config: SimulationConfig, schemeConfig: Any, surfaceConfig: SurfaceDetectionConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    xi = sphKernel_xi(config.kernel, config.dim)
    fs = torch.linalg.norm(colorGrad, dim = -1) > surfaceConfig.colorFieldGradThreshold * currentState.supports / xi
    return fs
