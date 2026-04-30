from typing import NamedTuple

from compressibleSPH.config import SimulationConfig
from compressibleSPH.utils import *
from sphWarpCore import *
import torch
from compSystem import *
from eos import idealGasEOS
from optimalSupport import evaluateOptimalSupport

class sodInitialState(NamedTuple):
    p: float
    rho: float
    v: float


def buildSod1D(
    nx: int,
    samplingRatio: int,
    leftState: sodInitialState,
    rightState: sodInitialState,
    gamma: float,
    config: SimulationConfig,
    smoothIC: bool = False
):
    
    actualRatio = nx / (nx // samplingRatio)

    particles_l = sampleRegularParticles(nx, buildDomainDescription(1, config.dim, periodic = True, device = config.device, dtype = config.dtype), config.targetNeighbors, jitter = 0.0)
    particles_r = sampleRegularParticles(nx // samplingRatio, buildDomainDescription(1, config.dim, periodic = True, device = config.device, dtype = config.dtype), config.targetNeighbors, jitter = 0.0)

    particles_r = particles_r._replace(masses = torch.ones_like(particles_r.masses) * particles_l.masses.min())
    pos_r = particles_r.positions
    pos_r[pos_r[:,0] < 0, 0] -= 0.5
    pos_r[pos_r[:,0] > 0, 0] += 0.5
    particles_r = particles_r._replace(positions = pos_r)

    print(f'Left particles: {particles_l.positions.shape[0]}, Right particles: {particles_r.positions.shape[0]}')

    combinedPositions = torch.cat([particles_l.positions, particles_r.positions], dim = 0)
    tags = torch.cat([
        torch.zeros(particles_l.positions.shape[0], dtype = torch.int32, device = particles_l.positions.device),
        torch.ones(particles_r.positions.shape[0], dtype = torch.int32, device = particles_r.positions.device)], dim = 0)

    # particles_r = particles_r._replace(masses = torch.ones_like(particles_r.masses) * particles_l.masses.min() * rightState.rho * actualRatio)
    # particles_l = particles_l._replace(masses = particles_l.masses * leftState.rho)


    leftMass = particles_l.masses.min() * leftState.rho
    rightMass = particles_l.masses.min() * rightState.rho * actualRatio



    combinedMasses = torch.where(tags == 0, leftMass, rightMass)
    combinedSupports = torch.where(tags[:, None] == 0, particles_l.supports.min(), particles_r.supports.min())[:,0]
    combinedDensities = torch.where(tags == 0, leftState.rho, rightState.rho)
    combinedVelocities = torch.where(tags[:, None] == 0, float(leftState.v), float(rightState.v))
    combinedKinds = torch.zeros_like(tags)
    combinedMaterials = tags
    combinedUIDs = torch.cat([
        torch.arange(particles_l.positions.shape[0], dtype = torch.int32, device = particles_l.positions.device),
        torch.arange(particles_r.positions.shape[0], dtype = torch.int32, device = particles_r.positions.device)], dim = 0)


    particleState = CompressibleState(
        positions = combinedPositions,
        velocities = combinedVelocities,
        supports = combinedSupports,
        masses = combinedMasses,
        densities = combinedDensities,

        kinds = combinedKinds,
        materials = combinedMaterials,
        UIDs = combinedUIDs,
        UIDcounter=combinedUIDs.max() + 1,

        internalEnergies=None,
        totalEnergies=None,
        entropies=None,
        pressures=None,
        soundspeeds=None,
        
        divergence=torch.zeros_like(combinedDensities),
        alpha0s=torch.ones_like(combinedDensities),
        alphas=torch.ones_like(combinedDensities)
    )

    # print(f'Left state: p={leftState.p}, rho={leftState.rho}, v={leftState.v}')
    # print(f'Right state: p={rightState.p}, rho={rightState.rho}, v={rightState.v}')
    # print(f'Actual sampling ratio: {combinedPositions.shape[0] / particles_l.positions.shape[0]}')

    # print(f'Initial particle count: {combinedPositions.shape[0]}')
    # print(f'Positions Shape: {combinedPositions.shape}')
    # print(f'Masses Shape: {combinedMasses.shape}')
    # print(f'Densities Shape: {combinedDensities.shape}')
    # print(f'Velocities Shape: {combinedVelocities.shape}')
    # print(f'Supports Shape: {combinedSupports.shape}')
    # print(f'Kinds Shape: {combinedKinds.shape}')


    densities = warpOperation(
        particleState, 
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Density,
            supportMode = SupportScheme.Gather,
            gradientMode = config.gradientMode,
            laplacianMode = config.laplacianMode,
        ),
        domain = config.domain,
    )

    rho_optimal, h_optimal, rhos_iter, supports_iter = evaluateOptimalSupport(particleState, config, supportScheme = SupportScheme.Gather)
    particleState.supports = h_optimal

    # rho_optimal = densities
    # h_optimal = particleState.supports

    eosRho = rho_optimal if smoothIC else particleState.densities
    P_initial = torch.where(particleState.materials == 0, leftState.p, rightState.p)

    u = 1 / (gamma - 1) * (P_initial / eosRho)

    if smoothIC:
        particleState.densities = eosRho
        dx = particles_l.positions[1,0] - particles_l.positions[0,0]
        x = torch.where(particleState.positions[:,0] > 0., particleState.positions[:,0] - 0.5, particleState.positions[:,0] + 0.5)
        ramp = torch.exp(x/dx) / (1 + torch.exp(x/dx))
        # ramp =  / (torch.exp(x/dx) + 1)
        ramped = lambda a, b, x: (a - b) / (torch.exp(x/dx) + 1) + b

        # u_max = u.max()
        # u_min = u.min()
        # u[mask] = u_min * (1 - ratio[mask]) + u_max * ratio[mask] if u_max < u_min else u_min * (1 - ratio[mask]) + u_max * ratio[mask]
        left_u = 1 / (gamma - 1) * (leftState.p / leftState.rho)
        right_u = 1 / (gamma - 1) * (rightState.p / rightState.rho)
        u = torch.where(particleState.positions[:,0] > 0., ramped(left_u, right_u, x), u)

        x = torch.where(particleState.positions[:,0] < 0., particleState.positions[:,0] + 0.5, particleState.positions[:,0] - 0.5)
        u = torch.where(particleState.positions[:,0] < 0., ramped(right_u, left_u, x), u)

    # A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = P_initial, rho = rho, gamma = gamma)
    A_, u_, P_, c_s = idealGasEOS(A = None, u = u, P = None, rho = eosRho, gamma = gamma)

    internalEnergy = u_
    kineticEnergy = torch.linalg.norm(particleState.velocities, dim = -1) **2/ 2
    totalEnergy = (internalEnergy + kineticEnergy) * particleState.masses

    particleState.internalEnergies = internalEnergy
    particleState.totalEnergies = totalEnergy
    particleState.pressures = P_
    particleState.soundspeeds = c_s
    particleState.entropies = A_

    adjacency = buildVerletList(particleState, 
                                domain = config.domain,
                                verletScale = 2**(1/config.dim), supportMode = config.supportMode)

    compressibleSystem = CompressibleSystem(
        state=particleState, 
        adjacency = adjacency, 
        domain = config.domain)

    return compressibleSystem