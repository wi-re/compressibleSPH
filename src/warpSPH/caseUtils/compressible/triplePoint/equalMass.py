from ....sample import *
import torch
from ....sample.compressible import setupBasicCompressibleInitialState
from ....modules import *
from sphWarpCore import *
from ....modules.timestep.compressible import computeTimestep
import math
import numpy as np
from warpSPH import *


def sampleTriplePointEqualMass(
    splitX, splitY, 
    rho_I, p_I,
    rho_II, p_II,
    rho_III, p_III,
    nxs, config, schemeConfig, extraData, SimulationState, SimulationSystem
    ):
    compressibleSystem = sampleRegionSystem(
        nxs = nxs,
        splitLineX = splitX,
        splitLineY = splitY,
        config = config,
        schemeConfig = schemeConfig,
        SimulationState = SimulationState,
        SimulationSystem = SimulationSystem,
    )

    particles = compressibleSystem.state

    rhoInitial = particles.densities.clone()
    Pi = particles.pressures.clone()

    regionI = torch.logical_or(particles.positions[:,0] <= 1.0, particles.positions[:,0] >= 13)
    regionII = torch.logical_and(~regionI, particles.positions[:,1].abs() >= 1.5)
    regionIII = torch.logical_and(~regionI, particles.positions[:,1].abs() < 1.5)

    rhoInitial[regionI] = rho_I
    Pi[regionI] = p_I
    rhoInitial[regionII] = rho_II
    Pi[regionII] = p_II
    rhoInitial[regionIII] = rho_III
    Pi[regionIII] = p_III

    A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = Pi, rho = rhoInitial, gamma = schemeConfig.gamma)
    # v_initial = torch.zeros_like(particles_l.positions)

    vInitial = torch.zeros_like(compressibleSystem.state.positions)
    internalEnergy = u_ 
    kineticEnergy = torch.linalg.norm(vInitial, dim = -1) **2/ 2
    totalEnergy = (internalEnergy + kineticEnergy) * compressibleSystem.state.masses

    compressibleSystem.state.internalEnergies = u_
    compressibleSystem.state.totalEnergies = totalEnergy
    compressibleSystem.state.pressures = P_
    compressibleSystem.state.soundspeeds = c_s
    compressibleSystem.state.velocities = vInitial
    compressibleSystem.state.densities = rhoInitial


    # from warpSPH.modules.timestep.compressible import computeTimestep
    config.dt = computeTimestep(compressibleSystem, config, schemeConfig, dt = None) #* 2/3
    # print(f"Computed timestep: {config.dt} s")

    # display(config)
    return compressibleSystem
