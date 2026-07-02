from ....sample.compressible import setupBasicCompressibleInitialState
import torch
import math

def sampleShearingNoh(vs, nx, config, schemeConfig, extraData, SimulationState, SimulationSystem):

    compressibleSystem = setupBasicCompressibleInitialState(nx, config, schemeConfig, SimulationState, SimulationSystem)

    v_initial_y = vs * torch.cos(2 * math.pi * compressibleSystem.state.positions[:,0])
    v_initial_x = torch.where(compressibleSystem.state.positions[:,0] < 0, 1.0, -1.0)

    v_initial = torch.stack([v_initial_x, v_initial_y], dim=1)

    compressibleSystem.state.velocities = v_initial

    return compressibleSystem