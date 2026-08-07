import torch
from warpSPHCore import *

from ..shifting.delta import computeDeltaShift

# def shuffleParticles(state, config, schemeConfig, shiftIters, jitterAmount = 0.01):
#     priorPositions = state.positions.clone()

#     state.positions[state.kinds == 0] += torch.randn_like(state.positions)[state.kinds == 0] * state.supports[state.kinds == 0].view(-1,1) * jitterAmount

#     adjacency = None
#     for i in range(shiftIters):
#         adjacency = buildVerletList(
#             state, 
#             config.domain, verletScale = config.verletScale, supportMode = SupportScheme.SuperSymmetric,
#             priorNeighborhood = adjacency,
#             verbose = False)
        
#         shiftVector, adjacency = computeDeltaShift(
#             currentState = state,
#             config = config,
#             schemeConfig = schemeConfig,
#             domain = config.domain,
#             adjacency = adjacency,
#         )
#         state.positions[state.kinds == 0] += shiftVector[state.kinds == 0]
    
#     newPositions = state.positions.clone()
#     state.positions = priorPositions
#     return newPositions


def shuffleParticles(state, config, schemeConfig, shiftIters, jitterAmount = 0.01):
    priorPositions = state.positions.clone()

    state.positions[state.kinds == 0] += torch.randn_like(state.positions)[state.kinds == 0] * state.supports[state.kinds == 0].view(-1,1) * jitterAmount

    initialDensities = state.densities.clone()

    adjacency = None
    for i in range(shiftIters):
        adjacency = buildVerletList(
            state, 
            config.domain, verletScale = config.verletScale, supportMode = SupportScheme.SuperSymmetric,
            priorNeighborhood = adjacency,
            verbose = False)
        state.densities = warpOperation(
            state,
            operationProperties = OperationProperties(
                operation=WarpOperation.Density,
                kernel = config.kernel, 
                supportMode = SupportScheme.Gather
            ),
            domain = config.domain,
            adjacency = None
        )
        
        shiftVector, adjacency = computeDeltaShift(
            currentState = state,
            config = config,
            schemeConfig = schemeConfig,
            domain = config.domain,
            adjacency = adjacency,
        )
        state.positions[state.kinds == 0] += 10. * shiftVector[state.kinds == 0]
        # print(f'Iteration {i+1}/{shiftIters}: max shift = {shiftVector[state.kinds == 0].norm(dim=1).max().cpu().item():.6g}')
    
    newPositions = state.positions.clone()
    state.positions = priorPositions
    state.densities = initialDensities
    return newPositions