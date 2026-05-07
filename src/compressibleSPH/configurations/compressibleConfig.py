from sphWarpCore.diffusion.viscosity import DiffusionParameters, ViscosityTerms
# from ..system import CompressibleSystem, CompressibleSystemUpdate
# from ..config import SimulationConfig
import torch

# from ..modules import *
from sphWarpCore import *

from dataclasses import dataclass, field
from typing import Optional
from ..enumTypes import AdaptiveSupportScheme, ViscositySwitch

def buildDefaultDiffusionParamsCompressibleSPH():
    diffusionParams = DiffusionParameters()
    diffusionParams.c_s = 1
    diffusionParams.C_l = 1
    diffusionParams.C_q = 0
    diffusionParams.Cu_l = 1
    diffusionParams.Cu_q = 0
    diffusionParams.K = 1.0
    diffusionParams.thermalConductivity = 0.5
    diffusionParams.viscosityTerm = ViscosityTerms.Price2012_98.value
    diffusionParams.thermalConducitiyTerm = ViscosityTerms.Price2008.value
    diffusionParams.scaleBeta = False
    diffusionParams.monaghanSwitch = True
    diffusionParams.correctXi = True
    
    return diffusionParams

@dataclass
class CompressibleSPHConfig:
    gamma: float = field(default=1.4, metadata={'description': 'Adiabatic index'})
    backgroundPressure: float = field(default=0.0, metadata={'description': 'Background pressure to prevent tensile instability'})

    rho0: float = field(default=1.0, metadata={'description': 'Reference density'})

    adaptiveSupportScheme: AdaptiveSupportScheme = field(default=AdaptiveSupportScheme.Monaghan, metadata={'description': 'Adaptive support scheme to use'})
    adaptiveSupportIterations: int = field(default=1, metadata={'description': 'Number of iterations for adaptive support scheme'})
    adaptiveSupportThreshold: float = field(default=1e-3, metadata={'description': 'Threshold for adaptive support scheme'})
    adaptiveSupportCorrections: bool = field(default=True, metadata={'description': 'Whether to apply corrections in the adaptive support scheme (grad-H terms)'})


    diffusionParams: DiffusionParameters = field(default_factory=buildDefaultDiffusionParamsCompressibleSPH)
    viscositySwitch: ViscositySwitch = field(default=ViscositySwitch.NoneSwitch, metadata={'description': 'Viscosity switch to use'})
    viscositySwitchLimitXi: bool = field(default=False, metadata={'description': 'Whether to limit the viscosity switch based on the xi parameter'})

    schemeName: str = field(default='Compressible SPH', metadata={'description': 'Name of the compressible SPH scheme to use'})