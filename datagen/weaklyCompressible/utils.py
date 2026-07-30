from sphWarpCore import *
from warpSPH import *
import torch
import numpy as np

from warpSPH.utils.naca import eval_naca



def buildPresetObstacles(
        maxExtent, offsetX, L, fillRatio, angle
):
    domainL = L / 2
    fillHeight = fillRatio * L

    obstacles = {
        'equilateralBottom': {
            'maxExtent': maxExtent,
            'offsetX': offsetX,
            'offsetY': -domainL + maxExtent/4,
            'aspectRatio': 2.0,
            'obstacleType': 'equilateralTriangle',
            'aoa': 0.0
        },
        'equilateralMiddle': {
            'maxExtent': maxExtent,
            'offsetX': offsetX,
            'offsetY': -domainL + maxExtent/2 + fillHeight/2 - maxExtent/2,
            'aspectRatio': 2.0,
            'obstacleType': 'equilateralTriangle',
            'aoa': angle # [0, 180]
        },
        'equilateralTop': {
            'maxExtent': maxExtent,
            'offsetX': offsetX,
            'offsetY': -domainL + fillHeight,
            'aspectRatio': 2.0,
            'obstacleType': 'equilateralTriangle',
            'aoa': angle # [-90, 0, 90, 180]
        },
        'triangleBottom': {
            'maxExtent': maxExtent/2,
            'offsetX': offsetX,
            'offsetY': -domainL + maxExtent/8,
            'aspectRatio': 1.0,
            'obstacleType': 'equilateralTriangle',
            'aoa': 0.0
        },
        'triangleMiddle': {
            'maxExtent': maxExtent/2,
            'offsetX': offsetX,
            'offsetY': -domainL + maxExtent/8 + fillHeight/2 - maxExtent/4,
            'aspectRatio': 1.0,
            'obstacleType': 'equilateralTriangle',
            'aoa': angle  # [-90, -45, -30, 0, 30, 45, 90]
        },
        'triangleTop': {
            'maxExtent': maxExtent/2,
            'offsetX': offsetX,
            'offsetY': -domainL + fillHeight,
            'aspectRatio': 1.0,
            'obstacleType': 'equilateralTriangle',
            'aoa': angle # [-90, -45, -30, 0, 30, 45, 90]
        },
        'circleBottom': {
            'maxExtent': maxExtent/2,
            'offsetX': offsetX,
            'offsetY': -domainL,
            'aspectRatio': 1.0,
            'obstacleType': 'circle',
            'aoa': 0.0 
        },
        'circleMiddle': {
            'maxExtent': maxExtent/2,
            'offsetX': offsetX,
            'offsetY': -domainL + fillHeight/2,
            'aspectRatio': 1.0,
            'obstacleType': 'circle',
            'aoa': 0.0 
        },
        'circleTop': {
            'maxExtent': maxExtent/2,
            'offsetX': offsetX,
            'offsetY': -domainL + fillHeight,
            'aspectRatio': 1.0,
            'obstacleType': 'circle',
            'aoa': 0.0
        },
        'ellipsoidBottom': {
            'maxExtent': maxExtent*0.5,
            'offsetX': offsetX,
            'offsetY': -domainL,
            'aspectRatio': 2.0,
            'obstacleType': 'ellipse',
            'aoa': 0.0
        },
        'ellipsoidMiddle': {
            'maxExtent': maxExtent*0.5,
            'offsetX': offsetX,
            'offsetY': -domainL + fillHeight/2,
            'aspectRatio': 2.0,
            'obstacleType': 'ellipse',
            'aoa': angle # [-90, -45, -30, 0, 30, 45, 90]
        },
        'ellipsoidTop': {
            'maxExtent': maxExtent*0.5,
            'offsetX': offsetX,
            'offsetY': -domainL + fillHeight,
            'aspectRatio': 2.0,
            'obstacleType': 'ellipse',
            'aoa': angle # [-90, -45, -30, 0, 30, 45, 90]
        },
        'squareBottom': {
            'maxExtent': maxExtent/2,
            'offsetX': offsetX,
            'offsetY': -domainL + maxExtent/3,
            'aspectRatio': 1.0,
            'obstacleType': 'box',
            'aoa': 0.0
        },
        'squareMiddle': {
            'maxExtent': maxExtent/3,
            'offsetX': offsetX,
            'offsetY': -domainL + fillHeight/2,
            'aspectRatio': 1.0,
            'obstacleType': 'box',
            'aoa': angle # [-45, -30, 0, 30]
        },
        'squareTop': {
            'maxExtent': maxExtent/2,
            'offsetX': offsetX,
            'offsetY': -domainL + fillHeight,
            'aspectRatio': 1.0,
            'obstacleType': 'box',
            'aoa': angle # [-45, -30, 0, 30]
        },
        'wallBottom' : {
            'maxExtent': maxExtent/2,
            'offsetX': offsetX,
            'offsetY': -domainL + maxExtent/2 - maxExtent / 3.0 * np.sin(np.abs(angle) * np.pi / 180),
            'aspectRatio': 3.0,
            'obstacleType': 'box',
            'aoa': 90.0 + angle # -45, -30, 0, 30, 45
        },
        'wallMiddle': {
            'maxExtent': maxExtent/2,
            'offsetX': offsetX,
            'offsetY': -domainL + fillHeight/2,
            'aspectRatio': 3.0,
            'obstacleType': 'box',
            'aoa': 90.0 + angle # -45, -30, 0, 30, 45, 90
        },
        'wallTop': {
            'maxExtent': maxExtent/2,
            'offsetX': offsetX,
            'offsetY': -domainL + fillHeight,#, - maxExtent/2 +  maxExtent / 3.0 * np.sin(np.abs(angle) * np.pi / 180),
            'aspectRatio': 3.0,
            'obstacleType': 'box',
            'aoa': 90.0 + angle # -45, -30, 0, 30, 45, 90
        }
    }
    return obstacles

def scaleFn(points, scaleX, scaleY):
    newPoints = points.clone()
    # return newPoints
    newPoints[:,0] = points[:,0] / scaleX
    newPoints[:,1] = points[:,1] / scaleY
    # print(f"Scaled points: {newPoints}")
    return newPoints

def translateFn(points, translateX, translateY):
    newPoints = points.clone()
    newPoints[:,0] = points[:,0] - translateX
    newPoints[:,1] = points[:,1] - translateY
    return newPoints

def rotateFn(points, angle):
    newPoints = points.clone()
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    newPoints[:,0] = points[:,0] * cos_angle - points[:,1] * sin_angle
    newPoints[:,1] = points[:,0] * sin_angle + points[:,1] * cos_angle
    return newPoints

def buildObstacleSDF(obstacleType, 
                     offsetX, offsetY, 
                     maxExtent, aspectRatio,
                     aoa, 
                     config, schemeConfig,  L, W = None):
    if W is None:
        W = L

# obstacleType = args.obstacleType

    # aoa = args.aoa
    aoa_rad = aoa / 180 * np.pi

    scale = 1.0

    trs = lambda points: scaleFn(rotateFn(translateFn(points, offsetX, offsetY), torch.tensor(aoa_rad).to(points.device)), scale, scale / aspectRatio)

    # circle
    if obstacleType == 'circle':
        obstacle_sdf = lambda x: getSDF('circle')['function'](trs(x), torch.tensor(maxExtent).to(x.device))
    # ellipse (emulated as a scaled circle)
    elif obstacleType == 'ellipse':
        obstacle_sdf = lambda x: getSDF('circle')['function'](trs(x), torch.tensor(maxExtent).to(x.device))
    # box
    elif obstacleType == 'box':
        obstacle_sdf = lambda x: getSDF('box')['function'](trs(x), torch.tensor([maxExtent,maxExtent]).to(x.device))
    # roundedBox
    elif obstacleType == 'roundedBox':
        obstacle_sdf = lambda x: getSDF('roundedBox')['function'](trs(x), torch.tensor([maxExtent,maxExtent ]).to(x.device), torch.tensor([maxExtent/5] * 4).to(x.device))
    # equilateralTriangle
    elif obstacleType == 'equilateralTriangle':
        obstacle_sdf = lambda x: getSDF('equilateralTriangle')['function'](trs(x), maxExtent)
    # hexagon
    elif obstacleType == 'hexagon':
        obstacle_sdf = lambda x: getSDF('hexagon')['function'](trs(x), torch.tensor(1/4).to(x.device))
    # horseshoe
    elif obstacleType == 'horseshoe':
        aperture = np.pi / 4
        obstacle_sdf = lambda x: getSDF('horseshoe')['function'](trs(x), torch.tensor([np.sin(aperture), np.cos(aperture)]).to(x.device), maxExtent*0.85, maxExtent/8)
    # star
    elif obstacleType == 'star':
        obstacle_sdf = lambda x: getSDF('star5')['function'](trs(x),maxExtent, maxExtent *1.25)
    # naca 4412
    # elif obstacleType.startswith('naca'):
    #     naca_id = obstacleType[4:]

    #     obstacle_sdf = lambda points: eval_naca(naca_id, scaleFn(rotateFn(translateFn(points, 0.0, 0.0), torch.tensor(aoa_rad).to(points.device)), scale, scale))
    #     tempRegion = buildRegion(config, schemeConfig, obstacle_sdf, RegionType.Boundary, initialConditions = {}, kind = BCType.noSlip, shortEdge = W > L)
    #     aabb = (torch.min(tempRegion.particles.positions, dim=0).values, torch.max(tempRegion.particles.positions, dim=0).values)
    #     # shift the airfoil so the vertical center of the bounding box is at y=0 and the leading edge is at x=offsetX
    #     new_offsetY = -(aabb[0][1] + aabb[1][1]) / 2
    #     new_offsetX = offsetX - aabb[0][0]
    #     obstacle_sdf = lambda points: eval_naca(naca_id, scaleFn(rotateFn(translateFn(points, new_offsetX, new_offsetY), torch.tensor(aoa_rad).to(points.device)), scale, scale))

    return obstacle_sdf


from warpSPH.utils.naca import *
# from utils import buildObstacleSDF
import copy


def build_sdfs(config, schemeConfig, band, args, domain, interiorDomain, obstacle):
    fluid_sdf = lambda x: sampleDomainSDF(x, domain, invert = True)



    union = lambda sdf1, sdf2: operatorDict['union'](sdf1, sdf2)

    interiorDomain2 = copy.deepcopy(interiorDomain)
    interiorDomain2.min[0] -= 20 * (band)
    interiorDomain2.max[0] += 20 * (band)
    domain_sdf = lambda x: sampleDomainSDF(x, interiorDomain2, invert = False)

    # obstacleWidth = L/16
    # obstacleHeight = L/4

    fluidW = args.W
    fluidH = args.L * args.fillRatio
    box_sdf = lambda points: sampleSDF(points, operatorDict['translate'](lambda x: getSDF('box')['function'](x, torch.tensor([fluidW/2,fluidH/2]).to(points.device)), torch.tensor([0.0,interiorDomain.min[1] + fluidH/2]).to(points.device)), invert = False)

    maxExtent = obstacle['maxExtent']
    aspectRatio = obstacle['aspectRatio']
    offsetX = obstacle['offsetX']
    offsetY = obstacle['offsetY']
    aoa = obstacle['aoa']

    domain_domain = copy.deepcopy(interiorDomain)

    if args.semiPeriodic:
        domain_domain.min[0] *= 1.5
        domain_domain.max[0] *= 1.5
    if args.fullyPeriodic:
        domain_domain.min[0] *= 1.5
        domain_domain.max[0] *= 1.5
        domain_domain.min[1] *= 1.5
        domain_domain.max[1] *= 1.5

    if args.obstacleActive:
        obstacle_sdf = buildObstacleSDF(obstacle['obstacleType'], offsetX, offsetY, maxExtent, aspectRatio, aoa, config, schemeConfig, args.L, args.W)
    domain_sdf = lambda x: domainSDF(x, domain_domain, invert = False)
    # obstacle_sdf = 

    if args.obstacleActive:
        merged_sdf = union(domain_sdf, obstacle_sdf)
    else:
        merged_sdf = domain_sdf
    domain_sdf = lambda x: sampleSDF(x, merged_sdf, invert=False)

    regions = []

    regions.append(buildRegion(config, schemeConfig, box_sdf, RegionType.Fluid, initialConditions = {}, shortEdge = args.W > args.L))
    # regions.append(buildRegion(config, schemeConfig, domain_sdf, RegionType.Boundary, initialConditions = {}, kind = BCType.noSlip))
    # if obstacle:

    # bcType = BCType.noSlip
    # if args.linearMotion or args.angularMotion:
    bcType = BCType.constant

    # if args.obstacleActive:
        # regions.append(buildRegion(config, schemeConfig, obstacle_sdf, RegionType.Boundary, initialConditions = {}, kind = bcType, shortEdge = W > L))
    # if args.band > 0:
    regions.append(buildRegion(config, schemeConfig, domain_sdf, RegionType.Boundary, initialConditions = {}, kind = BCType.noSlip, shortEdge = args.W > args.L))


    # fluid_sdf = lambda x: sampleDomainSDF(x, domain, invert = True)
    # regions = []

    # regions.append(buildRegion(config, schemeConfig, fluid_sdf, RegionType.Fluid, initialConditions = {}))

    for region in regions:
        region = filterRegion(region, regions)

    return regions, fluid_sdf, domain_sdf, obstacle_sdf if args.obstacleActive else None



def restoreConfig_from_h5(group, indent=0):
    config = {}
    # print('  '*indent + f'Restoring config from h5 group: {group.name}, keys: {list(group.keys())}, attributes: {list(group.attrs.keys())}')
    for key, value in group.attrs.items():
        if key != 'taggedType':
            # print('  '*indent + f'Restoring attribute for key: {key}, value: {value}')
            config[key] = value
    for key, subgroup in group.items():
        if subgroup.attrs['taggedType'] == 'dict':
            # print('  '*indent + f'Restoring dict for key: {key}, keys: {list(subgroup.keys())}')
            config[key] = restoreConfig_from_h5(subgroup, indent + 1)
        elif subgroup.attrs['taggedType'] == 'list':
            # print(f'Restoring list for key: {key}, length: {len(subgroup)}')
            config[key] = [restoreConfig_from_h5(subgroup[f'item_{i}'], indent + 1) for i in range(len(subgroup))]
            subkeys = list(subgroup.attrs.keys())
            subkeys = [k for k in subkeys if k.startswith('item_')]
            # print(f'Found {len(subkeys)} items in subgroup {key}: {subkeys}')
            for i in range(len(subkeys)):
                item_key = f'item_{i}'
                if item_key in subgroup.attrs:
                    # print(f'Restoring item_{i} for list key: {key}, value: {subgroup.attrs[item_key]}')
                    config[key].append(subgroup.attrs[item_key])
                # else:
                    # print(f'Warning: item_{i} not found in subgroup {key}')
            # print('  '*indent + f'Restored list for key: {key}, length: {len(config[key])} [{config[key]}]')
        else:
            print('  '*indent + f'Warning: Unknown type for subgroup {key}: {subgroup.attrs["taggedType"]}')
            raise ValueError(f'Unknown type for subgroup {key}: {subgroup.attrs["taggedType"]}')
    return config


# the config file contains the simulation parameters, which we copy to the compressed trajectory
# however, it is a multi-level dictionary, so we need to recursively copy the attributes
def copy_dict_to_h5(group, d, indent =0):
    # print('  '*indent + f'Copying dict to h5 group: {group.name}, keys: {list(d.keys())}')
    for key, value in d.items():
        if isinstance(value, dict):
            # print('  '*indent + f'Creating subgroup for key: {key}')
            subgroup = group.create_group(key)
            subgroup.attrs['taggedType'] = 'dict'
            copy_dict_to_h5(subgroup, value, indent + 1)
        elif isinstance(value, list):
            # print('  '*indent + f'Creating subgroup for key: {key}, value: {value}')
            subgroup = group.create_group(key)
            subgroup.attrs['taggedType'] = 'list'
            # if the value is a list of values that can be
            copy_dict_to_h5(subgroup, {f'item_{i}': v for i, v in enumerate(value)}, indent + 1)
        else:
            # print('  '*indent + f'Setting attribute for key: {key}, value: {value}')
            if value is None:
                pass
            else:
                group.attrs[key] = value
                
def restoreConfig_from_h5(group, indent=0):
    config = {}
    # print('  '*indent + f'Restoring config from h5 group: {group.name}, keys: {list(group.keys())}, attributes: {list(group.attrs.keys())}')
    for key, value in group.attrs.items():
        if key != 'taggedType':
            # print('  '*indent + f'Restoring attribute for key: {key}, value: {value}')
            config[key] = value
    for key, subgroup in group.items():
        if subgroup.attrs['taggedType'] == 'dict':
            # print('  '*indent + f'Restoring dict for key: {key}, keys: {list(subgroup.keys())}')
            config[key] = restoreConfig_from_h5(subgroup, indent + 1)
        elif subgroup.attrs['taggedType'] == 'list':
            # print(f'Restoring list for key: {key}, length: {len(subgroup)}')
            config[key] = [restoreConfig_from_h5(subgroup[f'item_{i}'], indent + 1) for i in range(len(subgroup))]
            subkeys = list(subgroup.attrs.keys())
            subkeys = [k for k in subkeys if k.startswith('item_')]
            # print(f'Found {len(subkeys)} items in subgroup {key}: {subkeys}')
            for i in range(len(subkeys)):
                item_key = f'item_{i}'
                if item_key in subgroup.attrs:
                    # print(f'Restoring item_{i} for list key: {key}, value: {subgroup.attrs[item_key]}')
                    config[key].append(subgroup.attrs[item_key])
                # else:
                    # print(f'Warning: item_{i} not found in subgroup {key}')
            # print('  '*indent + f'Restored list for key: {key}, length: {len(config[key])} [{config[key]}]')
        else:
            print('  '*indent + f'Warning: Unknown type for subgroup {key}: {subgroup.attrs["taggedType"]}')
            raise ValueError(f'Unknown type for subgroup {key}: {subgroup.attrs["taggedType"]}')
    return config



from dataclasses import dataclass
@dataclass
class SimulationProperties:
    device: torch.device
    dtype: torch.dtype

    nx: int
    dim: int
    L: float
    W: float

    dx: float
    band: int
    n_h: float
    targetDt: float
    freeSurface: bool
    semiPeriodic: bool
    fullyPeriodic: bool

    
def buildDomain(simSetup: SimulationProperties):
    device = simSetup.device
    dtype = simSetup.dtype
    domain = buildDomainDescription(simSetup.L + simSetup.dx * (simSetup.band) * 2, simSetup.dim, True, device, dtype)
    domain.min = torch.tensor([-simSetup.W/2 - simSetup.dx * (simSetup.band), -simSetup.L/2 - simSetup.dx * (simSetup.band)], device = device, dtype = dtype)
    domain.max = torch.tensor([simSetup.W/2 + simSetup.dx * (simSetup.band), simSetup.L/2 + simSetup.dx * (simSetup.band)], device = device, dtype = dtype)

    # Semi periodic
    if simSetup.semiPeriodic:
        domain.min = torch.tensor([-simSetup.W/2, -simSetup.L/2 - simSetup.dx * (simSetup.band)], device = device, dtype = dtype)
        domain.max = torch.tensor([simSetup.W/2, simSetup.L/2 + simSetup.dx * (simSetup.band)], device = device, dtype = dtype)

    # Closed domain
    if simSetup.fullyPeriodic:
        domain.min = torch.tensor([-simSetup.W/2, -simSetup.L/2], device = device, dtype = dtype)
        domain.max = torch.tensor([simSetup.W/2, simSetup.L/2], device = device, dtype = dtype)

    interiorDomain = buildDomainDescription(simSetup.L, simSetup.dim, False, device, dtype)
    interiorDomain.min = torch.tensor([-simSetup.W/2, -simSetup.L/2], device = device, dtype = dtype)
    interiorDomain.max = torch.tensor([simSetup.W/2, simSetup.L/2], device = device, dtype = dtype)
    return domain, interiorDomain


def buildRegions(config, schemeConfig, simSetup, args, domain, interiorDomain, obstacle):

    regions, fluid_sdf, domain_sdf, obstacle_sdf = build_sdfs(config, schemeConfig, args.band, args, domain, interiorDomain, obstacle)

    fluidW = args.fluidWidth * simSetup.W
    fluidH = args.fillRatio * simSetup.L

    # if args.semiPeriodic:
    #     fluidW = W*1.1
    # if args.fullyPeriodic:
    #     fluidW = W*1.1
    #     fluidH = L*1.1

    box_sdf = lambda points: sampleSDF(points, operatorDict['translate'](lambda x: getSDF('box')['function'](x, torch.tensor([fluidW/2,fluidH/2]).to(points.device)), torch.tensor([interiorDomain.min[0]+fluidW/2,interiorDomain.min[1] + fluidH/2]).to(points.device)), invert = False)

    # domain_sdf = lambda points: sampleSDF(points, domain_sdf, invert = False)

    regions = []
    regions.append(buildRegion(config, schemeConfig, domain_sdf, RegionType.Boundary, initialConditions = {}, kind = BCType.constant))
    # regions.append(buildRegion(config, schemeConfig, box_sdf, RegionType.Fluid, initialConditions = {}))

    # fluid_sdf = lambda x: sampleDomainSDF(x, domain, invert = True)
    # regions = []

    regions.append(buildRegion(config, schemeConfig, box_sdf, RegionType.Fluid, initialConditions = {}))

    for region in regions:
        region = filterRegion(region, regions)
        
    return regions


    
def sampleNoise(compressibleSystem, config, schemeConfig, simSetup, args):
    if args.enableNoise:
        velocities = sampleDivergenceFreeNoise(compressibleSystem.state, config.domain, config, schemeConfig, int(simSetup.nx * 2), 
                                            octaves = args.octaves, lacunarity = args.lacunarity, persistence = args.persistence, baseFrequency = args.baseFrequency, tileable = True, kind = args.kind, seed = args.seed)
        compressibleSystem.state.velocities[:] = velocities * args.noiseAmplitude 


def setupFreestream(compressibleSystem, config, schemeConfig, simSetup, args):        
    if args.enableFreestream:
        rho0 = schemeConfig.fluid.restDensity
        W = simSetup.W
        L = simSetup.L

        u_freestream = args.freeStreamVelocity
        forcingWidth = args.forcingWidth

        forcingSDF = lambda points: sampleSDF(points, lambda x: getSDF('box')['function'](x, torch.tensor([W/2 - forcingWidth, L/2]).to(points.device)), invert = True)

        def ldcDirichlet(state, cfg, schemeCfg, positions, d, n, t, dt):
            velocities = state.velocities.clone()
            # mask = torch.logical_or(positions[:,0] < -W/2 + forcingWidth, positions[:,0] > W/2 - forcingWidth)
            velocities[:,0] = u_freestream * 2
            # v_diff = u_freestream - state.velocities[:,0]
            # # slowly ramp the velocities to the target velocity over time
            # velocities[:,0] = state.velocities[:,0] + v_diff * dt / 0.1
            return velocities

        def ldcDirichletDensity(state, cfg, schemeCfg, positions, d, n, t, dt):
            densities = state.densities.clone()
            densities[:] = rho0
            # ramp the densities to the target density over time
            densities[:] = state.densities[:] + (rho0 - state.densities[:]) * dt / 0.1
            return densities

        def ldcDirichletUpdate(state, cfg, schemeCfg, positions, d, n, t, dt):
            velocities = torch.zeros_like(state.velocities)
            mask = torch.logical_or(positions[:,0] < -W/2 + forcingWidth, positions[:,0] > W/2 - forcingWidth)
            velocities[:,0] = torch.where(mask, 0.0, velocities[:,0])
            return velocities

        def ldcDirichletUpdateDensity(state, cfg, schemeCfg, positions, d, n, t, dt):
            densities = torch.zeros_like(state.densities)
            mask = torch.logical_or(positions[:,0] < -W/2 + forcingWidth, positions[:,0] > W/2 - forcingWidth)
            densities[:] = torch.where(mask, 0.0, densities[:])
            return densities

        def ldcForcing(state, cfg, schemeCfg, positions, d, n, t, dt):
            forcing = torch.zeros_like(state.velocities)
            # mask = torch.logical_or(positions[:,0] < -W/2 + forcingWidth, positions[:,0] > W/2 - forcingWidth)
            velocities = state.velocities.clone()
            velocities[:,0] = u_freestream
            v_diff = u_freestream - state.velocities[:,0]
            # slowly ramp the velocities to the target velocity over time
            velocities[:,0] = state.velocities[:,0] + v_diff * dt / 0.1

            forcing[:,0] = v_diff * dt / 0.1




            return forcing

        ldcBC = BoundaryCondition(
            type = BoundaryConditionType.dynamic,
            sdf = forcingSDF,
            dirichletFunctions = {
                # 'velocities': lambda state, cfg, schemeCfg, positions, d, n, t, dt: ldcDirichlet(state, cfg, schemeCfg, positions, d, n, t, dt),
                # 'densities': lambda state, cfg, schemeCfg, positions, d, n, t, dt: ldcDirichletDensity(state, cfg, schemeCfg, positions, d, n, t, dt)
            },
            updateFunctions = {
                # 'dvdt': lambda state, cfg, schemeCfg, positions, d, n, t, dt: ldcDirichletUpdate(state, cfg, schemeCfg, positions, d, n, t, dt),
                # 'drhodt': lambda state, cfg, schemeCfg, positions, d, n, t, dt: ldcDirichletUpdateDensity(state, cfg, schemeCfg, positions, d, n, t, dt)
            },
            forcingFunctions = [ldcForcing]

        )
        schemeConfig.boundaryConditions.append(ldcBC)
        minBoundaryDistance = torch.ones_like(compressibleSystem.state.positions[:,0]) * np.inf
        for region in schemeConfig.regions:
            if region.type == RegionType.Boundary:
                distances = region.sdf(compressibleSystem.state.positions)[0]
                minBoundaryDistance = torch.min(minBoundaryDistance, distances)
        maxDistance = compressibleSystem.state.supports.max() * 2.0
        minBoundaryDistance = torch.clamp(minBoundaryDistance, min=0.0, max = maxDistance)
        ramp = (minBoundaryDistance) / maxDistance
        def rampFn(r):
            ramped = 15/8 * r - 10/8 * r**3 + 3/8 * r**5
            return torch.clamp(ramped, min=0.0, max = 1.0)
        ramped = rampFn(ramp)

        compressibleSystem.state.velocities[compressibleSystem.state.kinds == 0,0] += u_freestream * ramped[compressibleSystem.state.kinds == 0]

def setupKolmogorov(compressibleSystem, config, schemeConfig, simSetup, args):        
    if args.enableKolmogorovForcing:
            
        xi = args.kolmogorovForcingAmplitude
        k = args.kolmogorovForcingWavenumber
        noiseLevel = 0.01 * xi
        from warpSPH.modules.noise.sampleDivergenceFree import generateNoiseInterpolator
        nxGrid = simSetup.nx * 2
        dim = simSetup.dim
        dtype = simSetup.dtype
        domain = config.domain
        device = config.device

        domain_cpu = buildDomainDescription(simSetup.L + simSetup.dx * (simSetup.band) * 2, dim, True, 'cpu', dtype)
        domain_cpu.min = torch.tensor([-simSetup.W/2 - simSetup.dx * (simSetup.band), -simSetup.L/2 - simSetup.dx * (simSetup.band)], device = 'cpu', dtype = dtype)
        domain_cpu.max = torch.tensor([simSetup.W/2 + simSetup.dx * (simSetup.band), simSetup.L/2 + simSetup.dx * (simSetup.band)], device = 'cpu', dtype = dtype)
        noiseGen = generateNoiseInterpolator(nxGrid, nxGrid, domain_cpu, dim = domain.dim, octaves = args.octaves, lacunarity = args.lacunarity, persistence = args.persistence, baseFrequency = args.baseFrequency, tileable = True, kind = args.kind, seed = args.seed)


        def forcing(state, config, compParams, x, d, n, t, dt):
        # def forcing(x, mask, state, t, dt):
            pos = getPeriodicPositions(x, domain)
            noiseOffset = torch.rand(pos.shape[1], device = device, dtype = dtype) * simSetup.L - simSetup.L/2
            noisePos = getPeriodicPositions(x + noiseOffset, domain)
            u_x = xi * torch.sin(k * np.pi * pos[:,1])
            u_y = noiseGen(pos.detach().cpu()).to(dtype = x.dtype, device = x.device) * noiseLevel
            return torch.stack([u_x, u_y], dim = 1) * state.masses.unsqueeze(1)

        kolmogorovForcing = BoundaryCondition(
            type = BoundaryConditionType.dynamic,
            sdf = lambda x: (torch.ones_like(x[:,0]) * -1.0, torch.zeros_like(x)),
            forcingFunctions = [forcing]
        )
        schemeConfig.boundaryConditions.append(kolmogorovForcing)