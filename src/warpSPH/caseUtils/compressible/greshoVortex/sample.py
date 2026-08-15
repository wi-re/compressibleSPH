"""Gresho-Chan rotating vortex initial state, used by `warpSPH.cases.greshoVortex`.

Samples a regular lattice and imposes the piecewise-analytic radial pressure
and angular-velocity profile of the vortex (three annuli, balanced so the
continuum solution is exactly steady), then derives internal energy, pressure,
and sound speed via `idealGasEOS` from the resulting density field.
"""

from ....sample import *
import torch
from ....sample.compressible import setupBasicCompressibleInitialState
from ....modules import *
from warpSPHCore import *
from ....modules.timestep.compressible import computeTimestep
import math

__all__ = ['sampleGreshoVortex']


def sampleGreshoVortex(nx, config, schemeConfig, SimulationState, SimulationSystem):
    compressibleSystem = setupBasicCompressibleInitialState(nx, config, schemeConfig, SimulationState, SimulationSystem)
    

    positions = compressibleSystem.state.positions
    r = torch.linalg.norm(positions, dim=-1)

    regionA = r < 0.2
    regionB = (r >= 0.2) & (r < 0.4)
    regionC = r >= 0.4  

    Pinitial = torch.zeros_like(compressibleSystem.state.densities)
    vinitial_angular = torch.zeros_like(compressibleSystem.state.densities)

    Pinitial[regionA] = 12.5 * r[regionA]**2 + 5
    Pinitial[regionB] = 12.5 * r[regionB]**2 - 20 * r[regionB] + 4 * torch.log(5*r[regionB]) + 9
    Pinitial[regionC] = 3 + 4 * math.log(2)

    vinitial_angular[regionA] = 5 * r[regionA]
    vinitial_angular[regionB] = 2 - 5 * r[regionB]
    vinitial_angular[regionC] = 0

    vinitial = torch.zeros_like(positions)
    vinitial[:,0] = -vinitial_angular * positions[:,1] / r
    vinitial[:,1] = vinitial_angular * positions[:,0] / r

    u = 1 / (schemeConfig.gamma - 1) * (Pinitial / compressibleSystem.state.densities)
    # A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = P_initial, rho = rho_optimal, gamma = gamma)
    A_, u_, P_, c_s = idealGasEOS(A = None, u = u, P = None, rho = compressibleSystem.state.densities, gamma = schemeConfig.gamma)

    compressibleSystem.state.velocities = vinitial
    compressibleSystem.state.internalEnergies = u_
    compressibleSystem.state.pressures = P_
    compressibleSystem.state.soundspeeds = c_s

    config.dt = computeTimestep(compressibleSystem, config, schemeConfig)
    return compressibleSystem