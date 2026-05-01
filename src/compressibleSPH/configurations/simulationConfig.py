import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from ..utils import *
from integrators import *
from sphWarpCore import *

from dataclasses import dataclass, field

# from waves.utils.domain import buildDomainDescription
from ..utils.sampling import SamplingScheme

@dataclass
class SimulationConfig:
    domain: DomainDescription = field(
        default_factory=lambda: buildDomainDescription(l=2, dim=2, periodic=True)
    )
    dim: int = 2
    kernel: KernelFunctions = KernelFunctions.Wendland2
    cflFactor: float = 0.3
    dt: Optional[float] = None
    adaptiveDt: bool = True
    targetNeighbors: int = field(default_factory=lambda: n_h_to_nH(4, 2))
    supportMode: SupportScheme = SupportScheme.SuperSymmetric
    gradientMode: GradientScheme = GradientScheme.Difference
    laplacianMode: LaplacianScheme = LaplacianScheme.Brookshaw
    integrationScheme: IntegrationSchemeType = IntegrationSchemeType.rungeKutta4
    samplingScheme: SamplingScheme = SamplingScheme.regular

    device: torch.device = field(default_factory=lambda: torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    dtype: torch.dtype = torch.float32
