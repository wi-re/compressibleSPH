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
from diffSPH.schemes.states.compressiblesph import CompressibleState as CompState
from diffSPH.kernels import getSPHKernelv2
from diffSPH.neighborhood import evaluateNeighborhood
from diffSPH.enums import KernelType as KernelTypeDiffSPH

from diffSPH.modules.compSPH import compSPH_acceleration, compSPH_dudt, compute_fij
from diffSPH.enums import EnergyScheme as EnergySchemeDiffSPH

def compSPH_step(
    system: CompSPHSystem,
    dt: float,
    config: SimulationConfig,
    compParams: CompSPHConfig,
    verbose = False,
):

    currentSystem = system#.initializeNewState()
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

    # diffSPHDiffusionConfig = {    
    # 'diffusion':{
    #     'C_l': 1,
    #     'C_q': 2,
    #     'Cu_l': 1,
    #     'Cu_q': 2,
    #     'monaghanSwitch': True,
    #     'viscosityTerm': 'Monaghan',
    #     'correctXi': True,
        
    #     'viscosityFormulation': 'Monaghan1992',
    #     'thermalConductivityFormulation': 'Monaghan1992',
    #     'signalTerm': 'Price2019',
    #     'K': 1.0,
        
    #     'thermalConductivity' : 0.5,
    # },
    # 'diffusionSwitch':{
    #     # 'scheme': ViscositySwitch.NoneSwitch,
    #     'limitXi': False,
    # },
    # 'domain': config.domain,
    # 'kernel': KernelTypeDiffSPH.Wendland2,
    # 'verbose': True,
    # }


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

    # diffSPHState = CompState(
    #     positions = currentState.positions,
    #     velocities = currentState.velocities,
    #     densities = currentState.densities,
    #     supports = currentState.supports,
    #     internalEnergies = currentState.internalEnergies,
    #     totalEnergies = currentState.totalEnergies,
    #     entropies = currentState.entropies,
    #     soundspeeds= currentState.soundspeeds,
    #     masses = currentState.masses,
    #     kinds = currentState.kinds,
    #     materials = currentState.materials,
    #     UIDs = currentState.UIDs,
    #     pressures = currentState.pressures,
    #     omega = gradHState.queryOmegas,
    # )

    # wrappedKernel = getSPHKernelv2(KernelTypeDiffSPH.Wendland2)
    # verletScale = 1
    # neighborhood, neighbors = evaluateNeighborhood(diffSPHState, config.domain, KernelTypeDiffSPH.Wendland2, verletScale = verletScale, mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)


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
    # dvdt = computePressureForceSymmetric(
    #     currentState,
    #     config,
    #     supportScheme = SupportScheme.KernelMeanSymmetric,
    #     adjacency = adjacency,
    #     gradH = gradHState
    # )
    diffusionParams = compParams.diffusionParams
    # dvdt += computeViscosity(
    #     currentState,
    #     # queryVelocities=currentState.velocities,
    #     operationProperties = OperationProperties(
    #         kernel = config.kernel,
    #         supportMode = SupportScheme.KernelMeanSymmetric,
    #     ),
    #     domain = config.domain,
    #     adjacency = adjacency,
    #     viscosityParams = diffusionParams,
    # )
    
    # dvdt, currentState.ap_ij, currentState.av_ij = compSPH_acceleration(diffSPHState, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, diffSPHDiffusionConfig)

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

    
    # dudt = computeDudtMonaghan(
    #     currentState,
    #     config,
    #     supportScheme = SupportScheme.KernelMeanSymmetric,
    #     adjacency = adjacency,
    #     gradH = gradHState
    # )
    # dudt += computeConductivity(
    #     currentState,
    #     # queryVelocities=currentState.velocities,
    #     operationProperties = OperationProperties(
    #         kernel = config.kernel,
    #         supportMode = SupportScheme.KernelMeanSymmetric,
    #     ),
    #     domain = config.domain,
    #     adjacency = adjacency,
    #     conductivityParams = diffusionParams,
    # )
    # dudt += computeThermalDissipation(
    #     currentState,
    #     # queryVelocities=currentState.velocities,
    #     operationProperties = OperationProperties(
    #         kernel = config.kernel,
    #         supportMode = SupportScheme.KernelMeanSymmetric,
    #     ),
    #     domain = config.domain,
    #     adjacency = adjacency,
    #     conductivityParams = diffusionParams,
    # )


    v_halfstep = currentState.velocities + 0.5 * dt * dvdt
    # diffSPHDiffusionConfig['energyScheme'] = EnergySchemeDiffSPH.CRK
    # currentState.f_ij = compute_fij(diffSPHState, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, diffSPHDiffusionConfig, config.dt, v_halfstep, currentState.ap_ij, currentState.av_ij)

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
    # dudt_diffSPH = compSPH_dudt(diffSPHState, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, diffSPHDiffusionConfig)

    # print(f'Max du/dt: {dudt.abs().max()}, Max du/dt diffSPH: {dudt_diffSPH.abs().max()}')
    # print(f'Max Diff: {(dudt - dudt_diffSPH).abs().max()}')

    drhodt = computeMomentumConsistent(
        currentState,
        config,
        supportScheme = SupportScheme.Gather,
        adjacency = adjacency,
        gradH = gradHState
    )
    dEdt = currentState.masses * torch.einsum('ij,ij->i', currentState.velocities, (dvdt)) + currentState.masses * (dudt)

    update = CompressibleSystemUpdate(
        dxdt = currentState.velocities.clone(),
        dvdt = dvdt,
        dudt = dudt,
        drhodt = drhodt,
        dEdt = dEdt,
    )

    return update, adjacency, currentState