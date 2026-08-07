
# from ..system import CompressibleSystem, CompressibleSystemUpdate
# from ..config import SimulationConfig
import torch

# from ..modules import *
from warpSPHCore import *

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
    pressureForceTerm: PressureForceScheme = field(default=PressureForceScheme.Antuono, metadata={'description': 'Pressure force term to use'})

    shiftProperties: ShiftProperties = field(default_factory=buildDefaultShiftProperties, metadata={'description': 'Properties for the delta-SPH shift'})

    regions: List[ParticleRegion] = field(default_factory=list, metadata={'description': 'List of particle regions in the simulation'})
    rigidBodies: List[RigidBody] = field(default_factory=list, metadata={'description': 'List of rigid bodies in the simulation'})

    surfaceDetectionConfig: SurfaceDetectionConfig = field(default_factory=buildDefaultSurfaceDetectionConfig, metadata={'description': 'Configuration for surface detection module'})

    gravityConfig: gravityConfiguration = field(default_factory=buildDefaultGravityConfiguration, metadata={'description': 'Configuration for gravity module'})

    bandwith: float = field(default=10.0, metadata={'description': 'Bandwith for the divergence-free noise sampling module'})

from typing import Dict, Any


def weaklyCompressibleConfigToDict(config: WeaklyCompressibleSPHConfig) -> Dict[str, Any]:
    return {
        'eosType': config.fluid.eosType.name,
        'restDensity': config.fluid.restDensity,
        'polytropicExponent': config.fluid.polytropicExponent,
        'kappa': config.fluid.kappa,
        'gas_constant': config.fluid.gas_constant,
        'molarMass': config.fluid.molarMass,
        'fixedSoundSpeed': config.fluid.fixedSoundSpeed if not isinstance(config.fluid.fixedSoundSpeed, torch.Tensor) else config.fluid.fixedSoundSpeed.detach().cpu().item(),

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
        'bandwith': config.bandwith,

        'pressureForceTerm': config.pressureForceTerm.name,
        'shiftProperties': {
            'iterations': config.shiftProperties.iterations,
            'CFL': config.shiftProperties.CFL,
            'computeMach': config.shiftProperties.computeMach,
            'maxC': config.shiftProperties.maxC,
            'active': config.shiftProperties.active,
            'scheme': config.shiftProperties.scheme.name,
            'projectionScheme': config.shiftProperties.projectionScheme.name,
            'summationDensity': config.shiftProperties.summationDensity,
            'surfaceScaling': config.shiftProperties.surfaceScaling,
            'threshold': config.shiftProperties.threshold,
            'projectQuantities': config.shiftProperties.projectQuantities,
        },
        'surfaceDetectionConfig': {
            'active': config.surfaceDetectionConfig.active,
            'colorFieldThreshold': config.surfaceDetectionConfig.colorFieldThreshold,
            'colorFieldGradThreshold': config.surfaceDetectionConfig.colorFieldGradThreshold,
            'barecascoThreshold': config.surfaceDetectionConfig.barecascoThreshold,
            'expansionIterations': config.surfaceDetectionConfig.expansionIterations,
            'scheme': config.surfaceDetectionConfig.scheme.name,
            'normalSource': config.surfaceDetectionConfig.normalSource.name,
        },
        'gravityConfig': gravityConfigurationToDict(config.gravityConfig),
        'regions': [region.toDict() for region in config.regions],
        'rigidBodies': [body.toDict() for body in config.rigidBodies],
    }

def dictToWeaklyCompressibleConfig(configDict: Dict[str, Any]) -> WeaklyCompressibleSPHConfig:
    config = WeaklyCompressibleSPHConfig()
    config.fluid.eosType = EquationOfState[configDict['eosType']] if isinstance(configDict['eosType'], str) else configDict['eosType']
    config.fluid.restDensity = float(configDict['restDensity'])
    config.fluid.polytropicExponent = float(configDict['polytropicExponent'])
    config.fluid.kappa = float(configDict['kappa'])
    config.fluid.gas_constant = float(configDict['gas_constant'])
    config.fluid.molarMass = float(configDict['molarMass'])
    config.fluid.fixedSoundSpeed = float(configDict['fixedSoundSpeed'])
    config.adaptiveSupportScheme = AdaptiveSupportScheme[configDict['adaptiveSupportScheme']] if isinstance(configDict['adaptiveSupportScheme'], str) else configDict['adaptiveSupportScheme']
    config.adaptiveSupportIterations = int(configDict['adaptiveSupportIterations'])
    config.adaptiveSupportThreshold = float(configDict['adaptiveSupportThreshold'])
    config.adaptiveSupportCorrections = bool(configDict['adaptiveSupportCorrections'])
    config.diffusionParams = dictToWCDiffusionParams(configDict['diffusionParams'])
    config.viscositySwitchParams = dictToViscositySwitchConfig(configDict['viscositySwitchParams'])
    config.schemeName = configDict['schemeName']
    config.boundaryConditions = [dictToBoundaryCondition(bcDict) for bcDict in configDict['boundaryConditions']]
    config.dt_viscosityConstraint = bool(configDict['dt_viscosityConstraint'])
    config.dt_accelerationConstraint = bool(configDict['dt_accelerationConstraint'])
    config.dt_acousticConstraint = bool(configDict['dt_acousticConstraint'])
    # config.densityDiffusionTerm = DensityDiffusionScheme[configDict['densityDiffusionTerm']] if isinstance(configDict['densityDiffusionTerm'], str) else configDict['densityDiffusionTerm']
    config.pressureForceTerm = PressureForceScheme[configDict['pressureForceTerm']] if isinstance(configDict['pressureForceTerm'], str) else configDict['pressureForceTerm']
    config.bandwith = float(configDict.get('bandwith', 10.0))
    shiftPropsDict = configDict.get('shiftProperties', {})
    config.shiftProperties = ShiftProperties(
        iterations=int(shiftPropsDict.get('iterations', 1)),
        CFL=float(shiftPropsDict.get('CFL', 0.3)),
        computeMach=bool(shiftPropsDict.get('computeMach', False)),
        maxC=float(shiftPropsDict.get('maxC', 0.3)),
        active=bool(shiftPropsDict.get('active', True)),
        scheme=ShiftingScheme[shiftPropsDict.get('scheme', ShiftingScheme.deltaSPH.name)] if isinstance(shiftPropsDict.get('scheme', ShiftingScheme.deltaSPH.name), str) else shiftPropsDict.get('scheme', ShiftingScheme.deltaSPH),
        projectionScheme=ShiftingProjectionScheme[shiftPropsDict.get('projectionScheme', ShiftingProjectionScheme.dot.name)] if isinstance(shiftPropsDict.get('projectionScheme', ShiftingProjectionScheme.dot.name), str) else shiftPropsDict.get('projectionScheme', ShiftingProjectionScheme.dot),
        summationDensity=bool(shiftPropsDict.get('summationDensity', False)),
        surfaceScaling=float(shiftPropsDict.get('surfaceScaling', 0.1)),
        threshold=float(shiftPropsDict.get('threshold', 0.5)),
        projectQuantities=bool(shiftPropsDict.get('projectQuantities', False)),
    )
    surfaceConfigDict = configDict.get('surfaceDetectionConfig')
    if surfaceConfigDict is not None:
        config.surfaceDetectionConfig = SurfaceDetectionConfig(
            active=surfaceConfigDict.get('active', buildDefaultSurfaceDetectionConfig().active),
            colorFieldThreshold=float(surfaceConfigDict.get('colorFieldThreshold', buildDefaultSurfaceDetectionConfig().colorFieldThreshold)),
            colorFieldGradThreshold=float(surfaceConfigDict.get('colorFieldGradThreshold', buildDefaultSurfaceDetectionConfig().colorFieldGradThreshold)),
            barecascoThreshold=float(surfaceConfigDict.get('barecascoThreshold', buildDefaultSurfaceDetectionConfig().barecascoThreshold)),
            expansionIterations=int(surfaceConfigDict.get('expansionIterations', buildDefaultSurfaceDetectionConfig().expansionIterations)),
            scheme=SurfaceDetectionScheme[surfaceConfigDict.get('scheme', buildDefaultSurfaceDetectionConfig().scheme.name)] if isinstance(surfaceConfigDict.get('scheme', buildDefaultSurfaceDetectionConfig().scheme.name), str) else surfaceConfigDict.get('scheme', buildDefaultSurfaceDetectionConfig().scheme),
            normalSource=NormalSource[surfaceConfigDict.get('normalSource', buildDefaultSurfaceDetectionConfig().normalSource.name)] if isinstance(surfaceConfigDict.get('normalSource', buildDefaultSurfaceDetectionConfig().normalSource.name), str) else surfaceConfigDict.get('normalSource', buildDefaultSurfaceDetectionConfig().normalSource),
        )
    config.gravityConfig = dictToGravityConfiguration(configDict['gravityConfig']) if configDict.get('gravityConfig') is not None else buildDefaultGravityConfiguration()
    config.regions = [ParticleRegion.fromDict(regionDict) for regionDict in configDict.get('regions', [])]
    config.rigidBodies = [RigidBody.fromDict(bodyDict) for bodyDict in configDict.get('rigidBodies', [])]

    return config