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
        domain_domain.min[0] *= 1.1
        domain_domain.max[0] *= 1.1
    if args.fullyPeriodic:
        domain_domain.min[0] *= 1.1
        domain_domain.max[0] *= 1.1
        domain_domain.min[1] *= 1.1
        domain_domain.max[1] *= 1.1

    obstacle_sdf = buildObstacleSDF(obstacle['obstacleType'], offsetX, offsetY, maxExtent, aspectRatio, aoa, config, schemeConfig, args.L, args.W)
    domain_sdf = lambda x: domainSDF(x, domain_domain, invert = False)
    # obstacle_sdf = 

    merged_sdf = union(domain_sdf, obstacle_sdf)
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

    return regions, fluid_sdf, domain_sdf, obstacle_sdf