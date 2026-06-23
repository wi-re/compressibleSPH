
# from ..system import CompressibleSystem, CompressibleSystemUpdate
# from ..config import SimulationConfig
import torch

# from ..modules import *
from sphWarpCore import *

from dataclasses import dataclass, field
from typing import Optional
from ..enumTypes import AdaptiveSupportScheme, ViscositySwitch, EquationOfState

from .diffusionParameters import DiffusionParameters, buildDefaultDiffusionParamsCompressibleSPH, diffusionParamsToDict, dictToDiffusionParams
from .viscositySwitchParameters import ViscositySwitchConfig, viscositySwitchConfigToDict, dictToViscositySwitchConfig




from .boundaryConditions import BoundaryCondition, BoundaryConditionType, boundaryConditionToDict, dictToBoundaryCondition
from typing import List


from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

@dataclass
class fluidProperties:
    eosType: EquationOfState = field(default=EquationOfState.isoThermal,metadata={"description": "Type of equation of state"})

    restDensity: float = field(default = 1.0, metadata={"description": "Rest density of the fluid"})

    polytropicExponent: Optional[float] = field(default=7.0, metadata={"description": "Polytropic exponent for polytropic EOS"})
    kappa: Optional[float] = field(default=1.3, metadata={"description": "Kappa"})
    gas_constant: Optional[float] = field(default=8.314, metadata={"description": "Gas constant"})
    molarMass: Optional[float] = field(default=0.02897, metadata={"description": "Molar mass of the gas"})

    fixedSoundSpeed: Optional[float] = field(default=10.0, metadata={"description": "Fixed sound speed"})

def buildDefaultFluidProperties() -> fluidProperties:
    return fluidProperties(
        eosType=EquationOfState.isoThermal,
        restDensity=1.0,
        polytropicExponent=7.0,
        kappa=1.3,
        gas_constant=8.314,
        molarMass=0.02897,
        fixedSoundSpeed=10.0
    )

from ..enumTypes import DensityDiffusionScheme, PressureForceScheme

@dataclass 
class ShiftProperties:
    iterations: int = field(default=1, metadata={"description": "Number of iterations for the delta-SPH shift"})
    CFL: float = field(default=0.3, metadata={"description": "CFL number for the delta-SPH shift"})
    computeMach: bool = field(default=False, metadata={"description": "Whether to compute Mach number for the delta-SPH shift"})
    maxC: float = field(default=0.3, metadata={"description": "Maximum sound speed for the delta-SPH shift"})
    active: bool = field(default=True, metadata={"description": "Whether to apply the delta-SPH shift"})

def buildDefaultShiftProperties() -> ShiftProperties:
    return ShiftProperties(
        iterations=1,
        CFL=0.3,
        computeMach=False,
        maxC=0.3,
        active=True
    )

@dataclass
class WeaklyCompressibleDiffusionParams():
    inviscid : bool = field(default=True, metadata={"description": "Whether to use inviscid diffusion parameters"})
    inviscidAlpha : float = field(default=0.01, metadata={"description": "Alpha value for inviscid diffusion"})

    viscidNu : float = field(default=1e-3, metadata={"description": "Kinematic viscosity for viscous diffusion"})

    densityDelta: float = field(default=0.1, metadata={"description": "Density diffusion coefficient for delta-SPH"})
    densityDiffusionTerm: DensityDiffusionScheme = field(default=DensityDiffusionScheme.deltaSPH, metadata={'description': 'Density diffusion term to use'})

def buildDefaultDiffusionParamsWeaklyCompressibleSPH() -> WeaklyCompressibleDiffusionParams:
    return WeaklyCompressibleDiffusionParams(
        inviscid=True,
        inviscidAlpha=0.01,
        viscidNu=1e-3,
        densityDelta=0.1,
        densityDiffusionTerm=DensityDiffusionScheme.deltaSPH
    )

@dataclass
class WeaklyCompressibleSPHConfig:
    fluid: fluidProperties = field(default_factory=buildDefaultFluidProperties, metadata={"description": "Fluid properties for the weakly compressible SPH simulation"})

    adaptiveSupportScheme: AdaptiveSupportScheme = field(default=AdaptiveSupportScheme.NoScheme, metadata={'description': 'Adaptive support scheme to use'})
    adaptiveSupportIterations: int = field(default=1, metadata={'description': 'Number of iterations for adaptive support scheme'})
    adaptiveSupportThreshold: float = field(default=1e-3, metadata={'description': 'Threshold for adaptive support scheme'})
    adaptiveSupportCorrections: bool = field(default=True, metadata={'description': 'Whether to apply corrections in the adaptive support scheme (grad-H terms)'})


    diffusionParams: WeaklyCompressibleDiffusionParams = field(default_factory=buildDefaultDiffusionParamsWeaklyCompressibleSPH, metadata={'description': 'Diffusion parameters for the weakly compressible SPH simulation'})

    viscositySwitchParams: ViscositySwitchConfig = field(default_factory=ViscositySwitchConfig)

    schemeName: str = field(default='Compressible SPH', metadata={'description': 'Name of the compressible SPH scheme to use'})

    boundaryConditions: List[BoundaryCondition] = field(default_factory=list, metadata={'description': 'List of boundary conditions to apply in the simulation'})

    dt_viscosityConstraint: bool = field(default=True, metadata={'description': 'Whether to apply viscosity constraint in timestep computation'})
    dt_accelerationConstraint: bool = field(default=True, metadata={'description': 'Whether to apply acceleration constraint in timestep computation'})
    dt_acousticConstraint: bool = field(default=True, metadata={'description': 'Whether to apply acoustic constraint in timestep computation'})
    pressureForceTerm: PressureForceScheme = field(default=PressureForceScheme.nonConservative, metadata={'description': 'Pressure force term to use'})

    shiftProperties: ShiftProperties = field(default_factory=buildDefaultShiftProperties, metadata={'description': 'Properties for the delta-SPH shift'})



from typing import Dict, Any


def wcDiffusionParamsToDict(diffusionParams: WeaklyCompressibleDiffusionParams) -> Dict[str, Any]:
    return {
        'inviscid': diffusionParams.inviscid,
        'inviscidAlpha': diffusionParams.inviscidAlpha,
        'viscidNu': diffusionParams.viscidNu,
        'densityDelta': diffusionParams.densityDelta,
        'densityDiffusionTerm': diffusionParams.densityDiffusionTerm.name if isinstance(diffusionParams.densityDiffusionTerm, Enum) else diffusionParams.densityDiffusionTerm
    }
def dictToWCDiffusionParams(diffusionParamsDict: Dict[str, Any]) -> WeaklyCompressibleDiffusionParams:
    return WeaklyCompressibleDiffusionParams(
        inviscid=diffusionParamsDict.get('inviscid', True),
        inviscidAlpha=diffusionParamsDict.get('inviscidAlpha', 0.01),
        viscidNu=diffusionParamsDict.get('viscidNu', 1e-3),
        densityDelta=diffusionParamsDict.get('densityDelta', 0.1),
        densityDiffusionTerm=DensityDiffusionScheme[diffusionParamsDict.get('densityDiffusionTerm', 'deltaSPH')] if isinstance(diffusionParamsDict.get('densityDiffusionTerm'), str) else diffusionParamsDict.get('densityDiffusionTerm', DensityDiffusionScheme.deltaSPH)
    )


def weaklyCompressibleConfigToDict(config: WeaklyCompressibleSPHConfig) -> Dict[str, Any]:
    return {
        'eosType': config.fluid.eosType.name,
        'restDensity': config.fluid.restDensity,
        'polytropicExponent': config.fluid.polytropicExponent,
        'kappa': config.fluid.kappa,
        'gas_constant': config.fluid.gas_constant,
        'molarMass': config.fluid.molarMass,
        'fixedSoundSpeed': config.fluid.fixedSoundSpeed,

        'adaptiveSupportScheme': config.adaptiveSupportScheme.name,
        'adaptiveSupportIterations': config.adaptiveSupportIterations,
        'adaptiveSupportThreshold': config.adaptiveSupportThreshold,
        'adaptiveSupportCorrections': config.adaptiveSupportCorrections,
        'diffusionParams': wcDiffusionParamsToDict(config.diffusionParams),
        'viscositySwitchParams': viscositySwitchConfigToDict(config.viscositySwitchParams),
        'schemeName': config.schemeName,
        'boundaryConditions': [boundaryConditionToDict(bc) for bc in config.boundaryConditions],
        'dt_viscosityConstraint': config.dt_viscosityConstraint,
        'dt_accelerationConstraint': config.dt_accelerationConstraint,
        'dt_acousticConstraint': config.dt_acousticConstraint,

        'pressureForceTerm': config.pressureForceTerm.name,
        'shiftProperties': {
            'iterations': config.shiftProperties.iterations,
            'CFL': config.shiftProperties.CFL,
            'computeMach': config.shiftProperties.computeMach,
            'maxC': config.shiftProperties.maxC,
            'active': config.shiftProperties.active
        },
    }

def dictToWeaklyCompressibleConfig(configDict: Dict[str, Any]) -> WeaklyCompressibleSPHConfig:
    config = WeaklyCompressibleSPHConfig()
    config.fluid.eosType = EquationOfState[configDict['eosType']] if isinstance(configDict['eosType'], str) else configDict['eosType']
    config.fluid.restDensity = configDict['restDensity']
    config.fluid.polytropicExponent = configDict['polytropicExponent']
    config.fluid.kappa = configDict['kappa']
    config.fluid.gas_constant = configDict['gas_constant']
    config.fluid.molarMass = configDict['molarMass']
    config.fluid.fixedSoundSpeed = configDict['fixedSoundSpeed']
    config.adaptiveSupportScheme = AdaptiveSupportScheme[configDict['adaptiveSupportScheme']] if isinstance(configDict['adaptiveSupportScheme'], str) else configDict['adaptiveSupportScheme']
    config.adaptiveSupportIterations = configDict['adaptiveSupportIterations']
    config.adaptiveSupportThreshold = configDict['adaptiveSupportThreshold']
    config.adaptiveSupportCorrections = configDict['adaptiveSupportCorrections']
    config.diffusionParams = dictToWCDiffusionParams(configDict['diffusionParams'])
    config.viscositySwitchParams = dictToViscositySwitchConfig(configDict['viscositySwitchParams'])
    config.schemeName = configDict['schemeName']
    config.boundaryConditions = [dictToBoundaryCondition(bcDict) for bcDict in configDict['boundaryConditions']]
    config.dt_viscosityConstraint = configDict['dt_viscosityConstraint']
    config.dt_accelerationConstraint = configDict['dt_accelerationConstraint']
    config.dt_acousticConstraint = configDict['dt_acousticConstraint']
    # config.densityDiffusionTerm = DensityDiffusionScheme[configDict['densityDiffusionTerm']] if isinstance(configDict['densityDiffusionTerm'], str) else configDict['densityDiffusionTerm']
    config.pressureForceTerm = PressureForceScheme[configDict['pressureForceTerm']] if isinstance(configDict['pressureForceTerm'], str) else configDict['pressureForceTerm']
    shiftPropsDict = configDict.get('shiftProperties', {})
    config.shiftProperties = ShiftProperties(
        iterations=shiftPropsDict.get('iterations', 1),
        CFL=shiftPropsDict.get('CFL', 0.3),
        computeMach=shiftPropsDict.get('computeMach', False),
        maxC=shiftPropsDict.get('maxC', 0.3),
        active=shiftPropsDict.get('active', True)
    )

    return config