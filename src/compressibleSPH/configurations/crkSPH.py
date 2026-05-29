from dataclasses import dataclass
import warp as wp
import torch

import warp as wp
from enum import Enum
from sphWarpCore.kernels.wp_kernel import sphKernel_xi
from sphWarpCore.mathutil import computeDistanceVec, safe_sqrt
import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any, Optional
import torch
from sphWarpCore.utils.wp_autograd import *

from sphWarpCore.radiusSearch.radius_util import AdjacencyList, AdjacencyListWarp, DomainDescription, PointCloud
from sphWarpCore.mathutil.wp_math import *
from sphWarpCore.kernels.wp_kernel import *

from sphWarpCore.enumTypes import *
# from sphWarpCore.util import *

from dataclasses import dataclass, field
from sphWarpCore.types import *


@wp.struct
class CRKViscosity:
    eta_fold: scalar_t = field(default = scalar_t(0.2))
    eta_crit: scalar_t = field(default = scalar_t(0.3333333))

    enableCRKLimiter: bool = field(default = True)
    enableVanLeerLimiter: bool = field(default = True)

    forceVanLeerOff: bool = field(default = False)
    forceVanLeerOn: bool = field(default = False)



from sphWarpCore.diffusion.viscosity import DiffusionParameters, ViscosityTerms
# from ..system import CompressibleSystem, CompressibleSystemUpdate
# from ..config import SimulationConfig
import torch
from ..enumTypes import EnergyScheme

# from ..modules import *
from sphWarpCore import *

from dataclasses import dataclass, field

from .compressibleConfig import CompressibleSPHConfig

def buildDefaultDiffusionParamsCRKSPH():
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
class CRKSPHConfig(CompressibleSPHConfig):
    energyScheme: EnergyScheme = field(default=EnergyScheme.CRK, metadata={'description': 'Energy scheme for the simulation'})

    diffusionParams: DiffusionParameters = field(default_factory=buildDefaultDiffusionParamsCRKSPH)
    crkViscosityParams: CRKViscosity = field(default_factory=CRKViscosity)
    schemeName: str = field(default='CRKSPH', metadata={'description': 'Name of the CRK SPH scheme to use'})