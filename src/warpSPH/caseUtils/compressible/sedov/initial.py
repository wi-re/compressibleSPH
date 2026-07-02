
from warpSPH.configurations.compressibleConfig import CompressibleSPHConfig
from warpSPH.modules.timestep.compressible import computeTimestep
from warpSPH.util import *
from warpSPH.schemes import *
from sphWarpCore.diffusion.viscosity import DiffusionParameters
from sphWarpCore import *
from sphWarpCore.radiusSearch.verlet import *
from sphWarpCore.radius import AdjacencyList
from sphWarpCore.operations import *
import torch
from sphWarpCore.enumTypes import *

from warp import Kernel

from warpSPH.configurations import SimulationConfig
from warpSPH.utils import *
from sphWarpCore import *
import torch
from warpSPH.systems import *
from warpSPH.modules import idealGasEOS, evaluateOptimalSupport
# from optimalSupport import evaluateOptimalSupport

from  warpSPH.enumTypes import AdaptiveSupportScheme
import warnings

from warpSPH.sample import *


def buildSedov(
        SimulationSystem, SimulationState,
        config: SimulationConfig,
        nx : int = 200,
        dim: int = 2,
        domainExtent: float = 2,
        periodicDomain: bool = True,

        rho0 : float = 1,
        E0 : float = 1,
        initialization: str = 'hat',
        gamma : float = 5/3,
        kernel: KernelFunctions = KernelFunctions.B7,
        targetNeighbors = 50,

        dtype = torch.float32,
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), shellSampling = False):
    # dim = 2
    domain = buildDomainDescription(domainExtent, dim, periodic = periodicDomain, device = device, dtype = dtype)

    if initialization == 'singular':
        if nx % 2 == 0:
            warnings.warn('nx should be odd for singular initialization, setting to nx + 1')
            nx += 1
    elif initialization == 'quadrant':
        if nx % 2 == 1:
            warnings.warn('nx should be even for quadrant initialization, setting to nx - 1')
            nx -= 1

    if shellSampling:
        particles_ = sampleShell(nx, domain, targetNeighbors)
    else:
        particles_ = sampleRegularParticles(nx, domain, targetNeighbors)
    # domain = buildDomainDescription(domainExtent * 1.5, dim, periodic = periodicDomain, device = device, dtype = dtype)
    particles_ = particles_._replace(masses = particles_.masses * rho0)

    

    particles = SimulationState(
        positions = particles_.positions,
        supports = particles_.supports,
        masses = particles_.masses,
        densities = particles_.densities,
        velocities = torch.zeros_like(particles_.positions),
        
        kinds = torch.zeros_like(particles_.positions[:,0], dtype = torch.int32),
        materials = torch.zeros_like(particles_.positions[:,0], dtype = torch.int32),
        UIDs = torch.arange(particles_.positions.shape[0], device = device, dtype = torch.int32),
        UIDcounter = particles_.positions.shape[0],
        
        internalEnergies = None,
        totalEnergies = None,
        entropies = None,
        pressures = None,
        soundspeeds = None,

        divergence=torch.zeros_like(particles_.densities),
        alpha0s= torch.ones_like(particles_.densities),
        alphas= torch.ones_like(particles_.densities),
    )

    
    densities = warpOperation(
        particles, 
        OperationProperties(
            kernel = kernel,
            operation = WarpOperation.Density,
            supportMode = SupportScheme.Gather,
            gradientMode = GradientScheme.Difference,
        ),
        domain = domain,
    )
    particles.densities = densities

    compressibleSPHConfig = CompressibleSPHConfig(
        adaptiveSupportIterations=16,
        adaptiveSupportThreshold=1e-3,
        adaptiveSupportScheme=AdaptiveSupportScheme.Owen,
        gamma=gamma
    )

    rho_optimal, h_optimal, adjacency, rhos_iter, supports_iter = evaluateOptimalSupport(particles, config, supportScheme = SupportScheme.Gather, compParams = compressibleSPHConfig)
    # particleState.supports = h_optimal

    particles.densities = rho_optimal
    particles.supports = h_optimal

    P_initial = torch.zeros_like(particles.densities)
    u = 1 / (gamma - 1) * (P_initial / rho_optimal)
    # A_, u_, P_, c_s = idealGasEOS(A = None, u = None, P = P_initial, rho = rho_optimal, gamma = gamma)
    A_, u_, P_, c_s = idealGasEOS(A = None, u = u, P = None, rho = rho_optimal, gamma = gamma)

    internalEnergy = u_ 
    kineticEnergy = torch.linalg.norm(torch.zeros_like(particles.positions), dim = -1) **2/ 2
    totalEnergy = (internalEnergy + kineticEnergy) * particles.masses

    simulationState_ = SimulationState(
        positions = particles.positions,
        supports = particles.supports,
        masses = particles.masses,
        densities = particles.densities,        
        velocities = torch.zeros_like(particles.positions),

        kinds = torch.zeros_like(particles_.positions[:,0], dtype = torch.int32),
        materials = torch.zeros_like(particles_.positions[:,0], dtype = torch.int32),
        UIDs = torch.arange(particles_.positions.shape[0], device = device, dtype = torch.int32),
        UIDcounter = particles.positions.shape[0],
        
        internalEnergies = u_,
        totalEnergies = totalEnergy,
        entropies = A_,
        pressures = P_,
        soundspeeds = c_s,

        alphas = torch.ones_like(particles.densities),
        alpha0s = torch.ones_like(particles.densities),
        divergence=torch.zeros_like(particles.densities),
    )
        

    if initialization == 'hat':
        # positions_warp = wp.from_torch(simulationState_.positions)
        raise NotImplementedError('Hat initialization not implemented yet')
        wrappedKernel = warpKernelToDiffSPHKernel(kernel)
        W = diffSPHKernel(wrappedKernel, torch.tensor([[0,0]], device = device, dtype = dtype) - simulationState_.positions, simulationState_.supports)
        W = W/W.sum()
        simulationState_.internalEnergies = E0 * W / simulationState_.masses
        A_, u_, P_, c_s = idealGasEOS(A = None, u = simulationState_.internalEnergies, P = None, rho = rho_optimal, gamma = gamma)
        simulationState_.pressures = P_
        simulationState_.soundspeeds = c_s
    elif initialization == 'singular':
        dist = torch.linalg.norm(simulationState_.positions, dim = -1)
        sortedDist, idx = torch.sort(dist)
        u_ = torch.zeros_like(simulationState_.masses)
        u_[idx[0]] = E0 / simulationState_.masses[idx[0]]

        # if sampleShell:
        #     u_[0] = E0 / particles.masses[0]
        # else:
        #     u_grid = u_.reshape(nx, nx)
        #     middle = nx // 2
        #     u_grid[middle, middle] = E0 / particles.masses[middle * nx + middle]
        #     u_ = u_grid.reshape(-1)
        A_, u_, P_, c_s = idealGasEOS(A = None, u = u_, P = None, rho = rho_optimal, gamma = gamma)
        simulationState_.internalEnergies = u_
        simulationState_.pressures = P_
        simulationState_.soundspeeds = c_s
    elif initialization == 'quadrant':
        if shellSampling:
            raise ValueError('Shell sampling not supported for quadrant initialization')
        if dim == 2: 
            nptcl = 4
        elif dim == 1:
            nptcl = 2
            
        dist = torch.linalg.norm(simulationState_.positions, dim = -1)
        sortedDist, idx = torch.sort(dist)
        u_ = torch.zeros_like(simulationState_.masses)
        u_[idx[:nptcl]] = E0 / simulationState_.masses[idx[:nptcl]] / nptcl
        


        # u_grid = u_.reshape(nx, nx)
        # middle = nx // 2
        # for ix in [middle - 1, middle]:
        #     for iy in [middle - 1, middle]:
        #         u_grid[ix, iy] = E0 / 4 / particles.masses[middle * nx + middle]
        # u_ = u_grid.reshape(-1)
        A_, u_, P_, c_s = idealGasEOS(A = None, u = u_, P = None, rho = rho_optimal, gamma = gamma)
        simulationState_.internalEnergies = u_
        simulationState_.pressures = P_
        simulationState_.soundspeeds = c_s
    else:
        raise ValueError(f'Unknown initialization {initialization}')

    adjacency = buildVerletList(simulationState_, 
                                domain = config.domain,
                                verletScale = 2**(1/config.dim), supportMode = config.supportMode)

    compressibleSystem = SimulationSystem(
        state=simulationState_, 
        adjacency = adjacency, 
        domain = config.domain)
    
    # particleSystem = SimulationSystem(
    #     systemState = simulationState_,
    #     domain = domain,
    #     neighborhoodInfo = neighborhood,
    #     t = 0
    # )

    dx = simulationState_.masses.min()
    config.dx = dx
    config.dt = computeTimestep(compressibleSystem, config, compressibleSPHConfig, dt = config.dt)
    return compressibleSystem
