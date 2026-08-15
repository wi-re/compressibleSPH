"""The classic Monaghan (1992-style) compressible SPH step: adaptive support
solve, super-symmetric adjacency, Gather-mode density, ideal-gas EOS, optional
grad-h corrections (`GradHState`), symmetric pressure force, Monaghan dudt,
and separate artificial-viscosity/conductivity/thermal-dissipation terms
added on top.
"""

# from warpSPH.modules import evaluateOptimalSupport, idealGasEOS, computeOmega
# from warpSPHCore import SupportScheme
# from warpSPH.modules import computePressureForceSymmetric, computeDudtMonaghan, computeMomentumConsistent
# from warpSPH.modules import computeViscosity, computeConductivity, computeThermalDissipation
from ..configurations.moduleConfigurations.diffusionParameters import DiffusionParameters, ViscosityTerms
from ..systems import CompressibleSystem, CompressibleSystemUpdate
from ..configurations import SimulationConfig, CompressibleSPHConfig
import torch

from ..modules.adaptiveSupport import computeOmega, evaluateOptimalSupport
from ..modules.boundaryConditions import computeForcing, enforceDirichlet, enforceUpdates
from ..modules.dissipation import computeConductivity, computeThermalDissipation, computeViscosity
from ..modules.eos import idealGasEOS
from ..modules.internalEnergy import computeDudtMonaghan
from ..modules.momentum import computeMomentumConsistent
from ..modules.pressure import computePressureForceSymmetric
from warpSPHCore import (
    GradHState, OperationProperties, SupportScheme,
    WarpOperation, buildVerletList, warpOperation,
)

__all__ = ['compressibleSPH_Monaghan']


def compressibleSPH_Monaghan(
    system: CompressibleSystem,
    dt: float,
    config: SimulationConfig,
    schemeConfig: CompressibleSPHConfig,
    verbose = False,
):
    currentSystem = system#.initializeNewState()
    currentState = currentSystem.state
    t = currentSystem.t

    rho_optimal, h_optimal, currentSystem.adjacency, *_ = evaluateOptimalSupport(currentState, config, schemeConfig, SupportScheme.Gather, currentSystem.adjacency)
    currentState.supports = h_optimal
    currentState.densities = rho_optimal

    # verletScale = 2 ** (1/config.dim)
    # verletScale = 1
    verletScale = config.verletScale

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

    enforceDirichlet(currentSystem, t, dt, config, schemeConfig)
    currentState.entropies, _, currentState.pressures, currentState.soundspeeds = idealGasEOS(
        A = None,
        u = currentState.internalEnergies,
        P = None,
        rho = currentState.densities,
        gamma = schemeConfig.gamma,
    )

    if schemeConfig.adaptiveSupportCorrections:
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
        schemeConfig = schemeConfig,
        adjacency = adjacency,
        gradH = gradHState
    )


    diffusionParams = schemeConfig.diffusionParams
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

    forcing = computeForcing(currentSystem, dt, t, config, schemeConfig)
    dvdt += forcing / currentState.masses.view(-1,1)


    update = CompressibleSystemUpdate(
        dxdt = currentState.velocities,
        dvdt = dvdt + dvdt_diss,
        dudt = dudt + dudt_diss + dudt_thermal,
        drhodt = drhodt,
        dEdt = dEdt,
    )
    enforceUpdates(update, currentSystem, dt, t, config, schemeConfig)

    return update, adjacency, currentState