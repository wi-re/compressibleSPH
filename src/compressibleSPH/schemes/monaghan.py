# from compressibleSPH.modules import evaluateOptimalSupport, idealGasEOS, computeOmega
# from sphWarpCore import SupportScheme
# from compressibleSPH.modules import computePressureForceSymmetric, computeDudtMonaghan, computeMomentumConsistent
# from compressibleSPH.modules import computeViscosity, computeConductivity, computeThermalDissipation
from sphWarpCore.diffusion.viscosity import DiffusionParameters
from ..system import CompressibleSystem, CompressibleSystemUpdate
from ..config import SimulationConfig, CompressibleSPHConfig
import torch

from ..modules import *
from sphWarpCore import *

def compressibleSPH_Monaghan(
    system: CompressibleSystem,
    dt: float,
    config: SimulationConfig,
    compParams: CompressibleSPHConfig,
    verbose = False,
):
    currentSystem = system.initializeNewState()
    currentState = currentSystem.state

    rho_optimal, h_optimal, currentSystem.adjacency, *_ = evaluateOptimalSupport(currentState, config, SupportScheme.Gather, currentSystem.adjacency)
    currentState.supports = h_optimal
    currentState.densities = rho_optimal

    verletScale = 2 ** (1/config.dim)
    # verletScale = 1

    adjacency = buildVerletList(
        currentState, 
        config.domain, verletScale = verletScale, supportMode = SupportScheme.SuperSymmetric,
        priorNeighborhood = currentSystem.adjacency,
        verbose = False)

    numNeighbors = adjacency.numNeighbors

    currentState.densities = warpOperation(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Density,
            supportMode = config.supportMode,
        ),
        domain = config.domain,
        adjacency = adjacency,
    )

    currentState.entropies, _, currentState.pressures, currentState.soundspeeds = idealGasEOS(
        A = None,
        u = currentState.internalEnergies,
        P = None,
        rho = currentState.densities,
        gamma = compParams.gamma,
    )

    omega = computeOmega(currentState, 
            OperationProperties(
                kernel = config.kernel,
                supportMode = SupportScheme.Gather,
            ),
            domain = config.domain,
            adjacency = adjacency
    )

    gradHState = GradHState(
        queryOmegas = omega
    )

    # from monaghanScheme import *

    dvdt = computePressureForceSymmetric(
        currentState,
        config,
        supportScheme = SupportScheme.KernelMeanSymmetric,
        adjacency = adjacency,
        gradH = gradHState
    )

    # currentState.velocities = torch.sin(currentState.positions[:,0]* np.pi).unsqueeze(-1)

    dudt = computeDudtMonaghan(
        currentState,
        config,
        supportScheme = SupportScheme.KernelMeanSymmetric,
        adjacency = adjacency,
        gradH = gradHState
    )

    drhodt = computeMomentumConsistent(
        currentState,
        config,
        supportScheme = SupportScheme.Gather,
        adjacency = adjacency,
        gradH = gradHState
    )


    diffusionParams = compParams.diffusionParams
    dvdt_diss = computeViscosity(
        currentState,
        # queryVelocities=currentState.velocities,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = SupportScheme.KernelMeanSymmetric,
        ),
        domain = config.domain,
        adjacency = adjacency,
        viscosityParams = diffusionParams,
    )


    dudt_diss = computeConductivity(
        currentState,
        # queryVelocities=currentState.velocities,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = SupportScheme.KernelMeanSymmetric,
        ),
        domain = config.domain,
        adjacency = adjacency,
        conductivityParams = diffusionParams,
    )


    dudt_thermal = computeThermalDissipation(
        currentState,
        # queryVelocities=currentState.velocities,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = SupportScheme.KernelMeanSymmetric,
        ),
        domain = config.domain,
        adjacency = adjacency,
        conductivityParams = diffusionParams,
    )

    dEdt = currentState.masses * torch.einsum('ij,ij->i', currentState.velocities, (dvdt + dvdt_diss)) + currentState.masses * (dudt + dudt_diss)

    update = CompressibleSystemUpdate(
        dxdt = currentState.velocities,
        dvdt = dvdt + dvdt_diss,
        dudt = dudt + dudt_diss + dudt_thermal,
        drhodt = drhodt,
        dEdt = dEdt,
    )

    return update, adjacency, currentState