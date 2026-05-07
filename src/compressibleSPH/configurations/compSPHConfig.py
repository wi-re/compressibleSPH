from sphWarpCore.diffusion.viscosity import DiffusionParameters
# from ..system import CompressibleSystem, CompressibleSystemUpdate
# from ..config import SimulationConfig
import torch
from ..enumTypes import EnergyScheme

# from ..modules import *
from sphWarpCore import *

from dataclasses import dataclass
@dataclass
class CompSPHConfig:
    gamma: float
    rho0: float

    diffusionParams: DiffusionParameters
    energyScheme: EnergyScheme = EnergyScheme.equalWork