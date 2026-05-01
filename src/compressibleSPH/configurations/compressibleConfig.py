from sphWarpCore.diffusion.viscosity import DiffusionParameters
# from ..system import CompressibleSystem, CompressibleSystemUpdate
# from ..config import SimulationConfig
import torch

# from ..modules import *
from sphWarpCore import *

from dataclasses import dataclass
@dataclass
class CompressibleSPHConfig:
    gamma: float
    rho0: float

    diffusionParams: DiffusionParameters