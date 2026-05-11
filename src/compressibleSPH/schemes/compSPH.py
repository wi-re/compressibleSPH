from ..modules.adaptiveSupport import computeOmega
from ..modules.compSPH.accel import computeCompSPHAccelWarp
from ..modules.compSPH.dudt import computeCompSPHdudtWarp
from ..modules.compSPH.balance import computeCompSPHBalanceTermWarp
from ..enumTypes import EnergyScheme
from ..modules import *

from sphWarpCore import *
from ..systems.compSPH import CompSPHSystem, CompSPHState
from ..configurations.compSPHConfig import CompSPHConfig
from ..configurations.simulationConfig import SimulationConfig
import torch
from ..systems.compressibleMonaghan import CompressibleSystemUpdate
# from diffSPH.schemes.states.compressiblesph import CompressibleState as CompState
# from diffSPH.kernels import getSPHKernelv2
# from diffSPH.neighborhood import evaluateNeighborhood
# from diffSPH.enums import KernelType as KernelTypeDiffSPH

# from diffSPH.modules.compSPH import compSPH_acceleration, compSPH_dudt, compute_fij
# from diffSPH.enums import EnergyScheme as EnergySchemeDiffSPH

from ..modules.shockCapturing.CullenHopkins import computeHopkinsTerms, computeHopkinsUpdate

def compSPH_step(
    system: CompSPHSystem,
    dt: float,
    config: SimulationConfig,
    compParams: CompSPHConfig,
    verbose = False,
):

    currentSystem = system#
    currentState = currentSystem.state

    rho_optimal, h_optimal, currentSystem.adjacency, *_ = evaluateOptimalSupport(currentState, config, compParams, SupportScheme.Gather, currentSystem.adjacency)
    currentState.supports = h_optimal
    currentState.densities = rho_optimal

    verletScale = config.verletScale

    adjacency = buildVerletList(
        currentState, 
        config.domain, verletScale = verletScale, supportMode = SupportScheme.SuperSymmetric,
        priorNeighborhood = currentSystem.adjacency,
        verbose = False)

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
    if currentState.divergence is None:
        drhodt = computeMomentumConsistent(
            currentState,
            config,
            supportScheme = SupportScheme.Gather,
            adjacency = adjacency,
            gradH = gradHState
        )
        currentState.divergence = drhodt

    currentState.entropies, _, currentState.pressures, currentState.soundspeeds = idealGasEOS(
        A = None,
        u = currentState.internalEnergies,
        P = None,
        rho = currentState.densities,
        gamma = compParams.gamma,
    )

    if compParams.adaptiveSupportCorrections:
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
    else:
        gradHState = None

    currentState.alphas, switchState = computeHopkinsTerms(
        currentState, 
        config, compParams, 
        SupportScheme.SuperSymmetric, 
        adjacency)   


    dvdt, currentState.ap_ij, currentState.av_ij = computeCompSPHAccelWarp(
        queryParticles = currentState,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode =  SupportScheme.KernelMeanSymmetric
        ),
        domain = config.domain,
        conductivityParams= compParams.diffusionParams,

        queryEnergies = currentState.internalEnergies,
        queryVelocities= currentState.velocities,
        queryCs = currentState.soundspeeds,
        queryAlphas = currentState.alphas,
        queryPressures = currentState.pressures,

        adjacency = adjacency,
        gradHState = gradHState
    )

    dudt = computeCompSPHdudtWarp(
        queryParticles = currentState,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = SupportScheme.KernelMeanSymmetric
        ),
        domain = config.domain,
        conductivityParams= compParams.diffusionParams,

        queryEnergies = currentState.internalEnergies,
        queryVelocities= currentState.velocities,
        queryCs = currentState.soundspeeds,
        queryAlphas = currentState.alphas,
        queryPressures = currentState.pressures,

        adjacency = adjacency,
        gradHState = gradHState
    )

    v_halfstep = currentState.velocities + 0.5 * dt * dvdt

    currentState.f_ij = computeCompSPHBalanceTermWarp(
        queryParticles = currentState,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = config.supportMode
        ),
        domain = config.domain,

        queryEnergies = currentState.internalEnergies,
        queryVelocities= v_halfstep,
        queryPressures = currentState.pressures,

        pairWise_pressureAccel= currentState.ap_ij,
        pairWise_viscosityAccel = currentState.av_ij,
        energyScheme = compParams.energyScheme,
        dt= dt.detach().cpu().item() if isinstance(dt, torch.Tensor) else dt,
        gamma = compParams.gamma,

        adjacency = adjacency,
        gradHState = gradHState
    )
    # particles.alpha0s, switchState = updateViscositySwitch(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config, dt, dvdt, switchState)

    currentState.alpha0s, switchState = computeHopkinsUpdate(
        switchState,
        dt, dvdt,
        currentState, 
        config, compParams, 
        SupportScheme.SuperSymmetric, 
        adjacency)   


    drhodt = computeMomentumConsistent(
        currentState,
        config,
        supportScheme = SupportScheme.Gather,
        adjacency = adjacency,
        gradH = gradHState
    )
    currentState.divergence = drhodt
    dEdt = currentState.masses * torch.einsum('ij,ij->i', currentState.velocities, (dvdt)) + currentState.masses * (dudt)

    update = CompressibleSystemUpdate(
        dxdt = currentState.velocities.clone(),
        dvdt = dvdt,
        dudt = dudt,
        drhodt = drhodt,
        dEdt = dEdt,
    )

    return update, adjacency, currentState