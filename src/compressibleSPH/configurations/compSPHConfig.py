from sphWarpCore.diffusion.viscosity import DiffusionParameters, ViscosityTerms
# from ..system import CompressibleSystem, CompressibleSystemUpdate
# from ..config import SimulationConfig
import torch
from ..enumTypes import EnergyScheme

# from ..modules import *
from sphWarpCore import *

from dataclasses import dataclass, field

from .compressibleConfig import CompressibleSPHConfig

def buildDefaultDiffusionParamsCompSPH():
    diffusionParams = DiffusionParameters()
        
    diffusionParams.c_s = 1
    diffusionParams.C_l = 1
    diffusionParams.C_q = 2
    diffusionParams.Cu_l = 1
    diffusionParams.Cu_q = 2
    diffusionParams.K = 1.0
    diffusionParams.thermalConductivity = 0.5
    diffusionParams.viscosityTerm = ViscosityTerms.Monaghan1992.value
    diffusionParams.thermalConducitiyTerm = ViscosityTerms.Price2012_98.value
    diffusionParams.scaleBeta = False
    diffusionParams.monaghanSwitch = True
    diffusionParams.correctXi = True
    
    return diffusionParams

@dataclass
class CompSPHConfig(CompressibleSPHConfig):
    energyScheme: EnergyScheme = field(default=EnergyScheme.CRK, metadata={'description': 'Energy scheme for the simulation'})

    diffusionParams: DiffusionParameters = field(default_factory=buildDefaultDiffusionParamsCompSPH)

    schemeName: str = field(default='CompSPH', metadata={'description': 'Name of the compressible SPH scheme to use'})

    compatibleEnergy: bool = field(default=True, metadata={'description': 'Whether to use a compatible energy discretization (e.g. evolve total energy and compute internal energy from it) or not (e.g. evolve internal energy directly)'})