
# from ..system import CompressibleSystem, CompressibleSystemUpdate
# from ..config import SimulationConfig
import torch

# from ..modules import *
from sphWarpCore import *

from dataclasses import dataclass, field
from typing import Optional
from ..enumTypes import AdaptiveSupportScheme, ViscositySwitch, EquationOfState

from .moduleConfigurations.diffusionParameters import DiffusionParameters, buildDefaultDiffusionParamsCompressibleSPH, diffusionParamsToDict, dictToDiffusionParams
from .moduleConfigurations.viscositySwitchParameters import ViscositySwitchConfig, viscositySwitchConfigToDict, dictToViscositySwitchConfig




from .moduleConfigurations.boundaryConditions import BoundaryCondition, BoundaryConditionType, boundaryConditionToDict, dictToBoundaryCondition
from typing import List


from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from .region import RegionType, ParticleRegion
from .rigidBody import RigidBody

from .moduleConfigurations.surfaceDetection import SurfaceDetectionConfig, buildDefaultSurfaceDetectionConfig
from .moduleConfigurations import *
from ..enumTypes import *

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

    regions: List[ParticleRegion] = field(default_factory=list, metadata={'description': 'List of particle regions in the simulation'})
    rigidBodies: List[RigidBody] = field(default_factory=list, metadata={'description': 'List of rigid bodies in the simulation'})

    surfaceDetectionConfig: SurfaceDetectionConfig = field(default_factory=buildDefaultSurfaceDetectionConfig, metadata={'description': 'Configuration for surface detection module'})


from typing import Dict, Any


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
        # 'surfaceDetectionConfig': surfaceDetectionConfigToDict(config.surfaceDetectionConfig) if config.surfaceDetectionConfig is not None else None
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