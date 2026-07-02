from ....sample.compressible import setupBasicCompressibleInitialState
import torch


def sampleNoh1D(nx, config, schemeConfig, SimulationState, SimulationSystem):
    particleSystem = setupBasicCompressibleInitialState(nx, config, schemeConfig, SimulationState, SimulationSystem)
    v_initial = - torch.nn.functional.normalize(particleSystem.state.positions, dim = -1)
    particleSystem.state.velocities = v_initial
    return particleSystem