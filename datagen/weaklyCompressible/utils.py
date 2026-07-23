from sphWarpCore import *
from warpSPH import *
import torch
import numpy as np

from warpSPH.utils.naca import eval_naca

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
        obstacle_sdf = lambda points: sampleSDF(points, lambda x: getSDF('circle')['function'](trs(x), torch.tensor(maxExtent).to(points.device)), invert = False)
    # ellipse (emulated as a scaled circle)
    elif obstacleType == 'ellipse':
        obstacle_sdf = lambda points: sampleSDF(scaleFn(points, maxExtent, maxExtent), lambda x: getSDF('circle')['function'](trs(x), torch.tensor(1.0).to(points.device)), invert = False)
    # box
    elif obstacleType == 'box':
        obstacle_sdf = lambda points: sampleSDF(points, lambda x: getSDF('box')['function'](trs(x), torch.tensor([maxExtent,maxExtent]).to(points.device)))
    # roundedBox
    elif obstacleType == 'roundedBox':
        obstacle_sdf = lambda points: sampleSDF(points, lambda x: getSDF('roundedBox')['function'](trs(x), torch.tensor([maxExtent,maxExtent ]).to(points.device), torch.tensor([maxExtent/5] * 4).to(points.device)), invert = False)
    # equilateralTriangle
    elif obstacleType == 'equilateralTriangle':
        obstacle_sdf = lambda points: sampleSDF(points, lambda x: getSDF('equilateralTriangle')['function'](trs(x), maxExtent), invert = False)
    # hexagon
    elif obstacleType == 'hexagon':
        obstacle_sdf = lambda points: sampleSDF(points, lambda x: getSDF('hexagon')['function'](trs(x), torch.tensor(1/4).to(points.device)), invert = False)
    # horseshoe
    elif obstacleType == 'horseshoe':
        aperture = np.pi / 4
        obstacle_sdf = lambda points: sampleSDF(points, lambda x: getSDF('horseshoe')['function'](trs(x), torch.tensor([np.sin(aperture), np.cos(aperture)]).to(points.device), maxExtent*0.85, maxExtent/8), invert = False)
    # star
    elif obstacleType == 'star':
        obstacle_sdf = lambda points: sampleSDF(points, lambda x: getSDF('star5')['function'](trs(x),maxExtent, maxExtent *1.25), invert = False)
    # naca 4412
    elif obstacleType.startswith('naca'):
        naca_id = obstacleType[4:]

        obstacle_sdf = lambda points: eval_naca(naca_id, scaleFn(rotateFn(translateFn(points, 0.0, 0.0), torch.tensor(aoa_rad).to(points.device)), scale, scale))
        tempRegion = buildRegion(config, schemeConfig, obstacle_sdf, RegionType.Boundary, initialConditions = {}, kind = BCType.noSlip, shortEdge = W > L)
        aabb = (torch.min(tempRegion.particles.positions, dim=0).values, torch.max(tempRegion.particles.positions, dim=0).values)
        # shift the airfoil so the vertical center of the bounding box is at y=0 and the leading edge is at x=offsetX
        new_offsetY = -(aabb[0][1] + aabb[1][1]) / 2
        new_offsetX = offsetX - aabb[0][0]
        obstacle_sdf = lambda points: eval_naca(naca_id, scaleFn(rotateFn(translateFn(points, new_offsetX, new_offsetY), torch.tensor(aoa_rad).to(points.device)), scale, scale))

    return obstacle_sdf
