import torch
from sphWarpCore import *

from ..shifting.delta import computeDeltaShift

def shuffleParticles(state, config, schemeConfig, shiftIters, jitterAmount = 0.01):
    priorPositions = state.positions.clone()

    state.positions[state.kinds == 0] += torch.randn_like(state.positions)[state.kinds == 0] * state.supports[state.kinds == 0].view(-1,1) * jitterAmount

    adjacency = None
    for i in range(shiftIters):
        shiftVector, adjacency = computeDeltaShift(
            currentState = state,
            config = config,
            schemeConfig = schemeConfig,
            domain = config.domain,
            adjacency = adjacency,
        )
        state.positions[state.kinds == 0] += shiftVector[state.kinds == 0]
    
    newPositions = state.positions.clone()
    state.positions = priorPositions
    return newPositions