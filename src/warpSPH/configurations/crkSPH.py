from dataclasses import dataclass
import warp as wp
import torch

import warp as wp
from enum import Enum
from sphWarpCore import *
import warp as wp
from warp.types import vector, matrix
# from wp_tensor import tensor
from typing import Any, Optional
import torch
from dataclasses import dataclass, field


@wp.struct
class CRKViscosity:
    eta_fold: scalar_t = field(default = scalar_t(0.2))
    eta_crit: scalar_t = field(default = scalar_t(0.3333333))

    enableCRKLimiter: bool = field(default = True)
    enableVanLeerLimiter: bool = field(default = True)

    forceVanLeerOff: bool = field(default = False)
    forceVanLeerOn: bool = field(default = False)

def buildDefaultCRKViscosityParams():
    crkViscosityParams = CRKViscosity()
    crkViscosityParams.eta_fold = 0.2
    crkViscosityParams.eta_crit = 0.3333333
    crkViscosityParams.enableCRKLimiter = True
    crkViscosityParams.enableVanLeerLimiter = True
    crkViscosityParams.forceVanLeerOff = False
    crkViscosityParams.forceVanLeerOn = False
    
    return crkViscosityParams


from .moduleConfigurations.diffusionParameters import DiffusionParameters, ViscosityTerms
# from ..system import CompressibleSystem, CompressibleSystemUpdate
# from ..config import SimulationConfig
import torch
from ..enumTypes import EnergyScheme

# from ..modules import *
from sphWarpCore import *

from dataclasses import dataclass, field

from .compressibleConfig import CompressibleSPHConfig, compressibleConfigToDict, dictToCompressibleConfig

def buildDefaultDiffusionParamsCRKSPH():
    diffusionParams = DiffusionParameters()
        
    diffusionParams.c_s = 1
    diffusionParams.C_l = 1
    diffusionParams.C_q = 1
    diffusionParams.Cu_l = 1
    diffusionParams.Cu_q = 1
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
    crkViscosityParams: CRKViscosity = field(default_factory=buildDefaultCRKViscosityParams)
    schemeName: str = field(default='CRKSPH', metadata={'description': 'Name of the CRK SPH scheme to use'})
    
    compatibleEnergy: bool = field(default=True, metadata={'description': 'Whether to use a compatible energy discretization (e.g. evolve total energy and compute internal energy from it) or not (e.g. evolve internal energy directly)'})

from typing import Dict, Any

def crkSPHConfigToDict(config: CRKSPHConfig) -> Dict[str, Any]:
    baseDict = compressibleConfigToDict(config)
    baseDict.update({
        'energyScheme': config.energyScheme.name,
        'compatibleEnergy': config.compatibleEnergy,
        'crkViscosityParams': {
            'eta_fold': config.crkViscosityParams.eta_fold,
            'eta_crit': config.crkViscosityParams.eta_crit,
            'enableCRKLimiter': config.crkViscosityParams.enableCRKLimiter,
            'enableVanLeerLimiter': config.crkViscosityParams.enableVanLeerLimiter,
            'forceVanLeerOff': config.crkViscosityParams.forceVanLeerOff,
            'forceVanLeerOn': config.crkViscosityParams.forceVanLeerOn
        }
    })
    return baseDict

def dictToCRKSPHConfig(configDict: Dict[str, Any]) -> CRKSPHConfig:
    compressibleConfig = dictToCompressibleConfig(configDict)
    crkSPHConfig = CRKSPHConfig(**compressibleConfig.__dict__)
    crkSPHConfig.energyScheme = EnergyScheme[configDict['energyScheme']] if isinstance(configDict['energyScheme'], str) else configDict['energyScheme']
    crkSPHConfig.compatibleEnergy = configDict['compatibleEnergy']
    crkViscosityParamsDict = configDict['crkViscosityParams']
    crkSPHConfig.crkViscosityParams = CRKViscosity()
    crkSPHConfig.crkViscosityParams.eta_fold = crkViscosityParamsDict['eta_fold']
    crkSPHConfig.crkViscosityParams.eta_crit = crkViscosityParamsDict['eta_crit']
    crkSPHConfig.crkViscosityParams.enableCRKLimiter = crkViscosityParamsDict['enableCRKLimiter']
    crkSPHConfig.crkViscosityParams.enableVanLeerLimiter = crkViscosityParamsDict['enableVanLeerLimiter']
    crkSPHConfig.crkViscosityParams.forceVanLeerOff = crkViscosityParamsDict['forceVanLeerOff']
    crkSPHConfig.crkViscosityParams.forceVanLeerOn = crkViscosityParamsDict['forceVanLeerOn']
    
    return crkSPHConfig