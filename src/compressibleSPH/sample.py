import torch
import numpy as np
from .utils import *
from typing import List, Union
from sphWarpCore import *

def smoothValuesWarp(quantity, particleState, nIters, neighbors, config):
    sampled  = quantity.clone()
    for _ in range(nIters):
        sampled = sphOperation_warp(
            queryPositions=particleState.positions, referencePositions=particleState.positions,
            querySupports=particleState.supports, referenceSupports=particleState.supports,
            queryMasses=particleState.masses, referenceMasses=particleState.masses,
            queryDensities = particleState.densities, referenceDensities = particleState.densities,
            queryValues = sampled, referenceValues = sampled,
            kernel = KernelFunctions.Wendland2,
            adjacency= neighbors, domain = config.domain,
            operation = WarpOperation.Interpolate,
            supportMode = SupportScheme.Gather)
    return sampled

import copy
def smoothState(state, particleState, smoothIters, neighbors, config):
    smoothState = (      smoothValuesWarp(state.u, particleState, smoothIters, neighbors, config),
        smoothValuesWarp(state.v, particleState, smoothIters, neighbors, config),
        smoothValuesWarp(state.c, particleState, smoothIters, neighbors, config),
        smoothValuesWarp(state.damping, particleState, smoothIters, neighbors, config)
    )    
    newState = copy.deepcopy(state)
    newState.u = smoothState[0]
    newState.v = smoothState[1]
    newState.c = smoothState[2]
    newState.damping = smoothState[3]
    return newState

# from waves.sample import sampleVoronoi, smoothValues
import math

def addNoise(
    particleState, config, neighbors,
    grid, cSourceGrid, noiseAmplitude = 0.1, uMagnitude = 10,
    noiseType: str = 'perlin',
    smoothIter: int = 4,
    seed: int = 42,
    octaves: int = 2,
    baseFrequency: int = 2,
):
    u_min = torch.min(grid).cpu().item()
    u_max = torch.max(grid).cpu().item()

    nx = int(math.sqrt(particleState.positions.shape[0]))
    if u_min == u_max:
        u_min = -uMagnitude
        u_max = uMagnitude

    generator = torch.Generator(device=particleState.positions.device)
    generator.manual_seed(seed)

    if noiseType == 'perlin':
        uNoise = sampleVoronoi(particleState.positions, nx * 2, octaves = octaves, baseFrequency = baseFrequency, seed = seed, config = config)
    elif noiseType == 'uniform':
        # uNoise = torch.rand_like(grid, generator=generator)
        uNoise = torch.rand_like(grid)
    elif noiseType == 'normal':
        # uNoise = torch.randn_like(grid, generator=generator)
        uNoise = torch.randn_like(grid)
    else:
        raise ValueError(f"Unsupported noise type: {noiseType}")

    uNoise = smoothValuesWarp(
        uNoise,
        particleState,
        smoothIter, neighbors,
        config
    )
    uNoiseNormalized = (uNoise - torch.min(uNoise)) / (torch.max(uNoise) - torch.min(uNoise))
    uNoise = uNoiseNormalized * (u_max - u_min) + u_min

    grid_lerp = torch.lerp(grid, uNoise, noiseAmplitude)

    # grid[~(cSourceGrid == -1)] = grid_lerp[~(cSourceGrid == -1)]
    grid[(cSourceGrid == 0)] = grid_lerp[(cSourceGrid == 0)]

    return grid


def populateCGrid(cGrid, cSourceGrid, 
        boundaryC = 0.01, obstacleC = 0.5, defaultC = 1.0,
        randomObstacleC = False, obstacleCRange = (0.3, 0.7)):
    cGrid = torch.ones_like(cGrid) * defaultC

    boundaryIds = torch.unique(cSourceGrid)
    boundaryIds = boundaryIds[boundaryIds != 0]  # Exclude background (0)

    for bid in boundaryIds:
        mask = (cSourceGrid == bid)
        if bid == -1:
            cGrid[mask] = boundaryC
        else:
            if randomObstacleC:
                cGrid[mask] = torch.empty_like(cGrid[mask]).uniform_(*obstacleCRange)
            else:
                cGrid[mask] = obstacleC
    return cGrid

def populateUGrid(uGrid, uSourceGrid, sourceMagnitudes : Union[float, int, List[float]], randomMagnitude = False, magnitudeRange = (-10.0, 10.0)):
    sourceIds = torch.unique(uSourceGrid)
    sourceIds = sourceIds[sourceIds != 0]  # Exclude background (0)

    for sid in sourceIds:
        # print('setting source id', sid)
        mask = (uSourceGrid == sid)
        sourceMagnitude = 0.0
        if randomMagnitude:
            sourceMagnitude = torch.empty(1).uniform_(*magnitudeRange).item()
        elif isinstance(sourceMagnitudes, float) or isinstance(sourceMagnitudes, int):
            sourceMagnitude = sourceMagnitudes
        else:
            sourceMagnitude = sourceMagnitudes[int(sid.item())-1]
        uGrid[mask] = sourceMagnitude
    return uGrid



def generateInitialVariables(nx, device = None):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    L = 2
    dim = 2

    kernel = KernelFunctions.Wendland4
    targetNeighbors = n_h_to_nH(4, dim)

    domain = buildDomainDescription(L, dim, True, device, dtype)

    # particles = sampleOptimal(nx, domain, targetNeighbors, kernel, 0., 0, shiftScheme = 'delta')

    config = {
        'domain': domain,
        'kernel': kernel,
        'targetNeighbors': targetNeighbors,
        'neighborhood':{
            'verletScale': 1.0
        }
    }

    # config['gradientMode'] = GradientMode.Difference
    # config['laplacianMode'] = LaplacianMode.Brookshaw
    # config['supportScheme'] = SupportScheme.Gather
    # config['integrationScheme'] = IntegrationSchemeType.rungeKutta4
    
    return config, domain, device, dtype, kernel