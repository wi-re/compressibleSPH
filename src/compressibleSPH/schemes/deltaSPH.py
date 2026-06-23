from ..configurations import *
from sphWarpCore.enumTypes import ViscosityTerms
from sphWarpCore.kernels.wp_kernel import sphKernel_xi
from ..systems import *
from ..modules import *
from ..enumTypes import *
from sphWarpCore import *


def deltaSPH_step(
    system: CompSPHSystem,
    dt: float,
    config: SimulationConfig,
    schemeConfig: WeaklyCompressibleSPHConfig,
    verbose = False,        
):
    currentSystem = system#
    currentState = currentSystem.state
    adjacency = currentSystem.adjacency

    # 1. Compute adjacency
    verletScale = config.verletScale

    adjacency = buildVerletList(
        currentState, 
        config.domain, verletScale = verletScale, supportMode = SupportScheme.SuperSymmetric,
        priorNeighborhood = adjacency,
        verbose = False)
    currentSystem.adjacency = adjacency

    # 2. Compute density if density is none
    if currentState.densities is None:
        print("Computing densities...")
        currentState.densities = computeDensities(currentState, config, schemeConfig, adjacency)
    # 3. Skipped mDBC density computation since no boundaries are present
    # 4. enforce BCs
    enforceDirichlet(currentSystem, currentSystem.t, config.dt, config, schemeConfig)
    # 5. compute EOS (WC version)
    currentState.pressures = weaklyCompressibleEOS(currentState, schemeConfig)
    # 6. Skipped boundary velocity computation since no boundaries are present
    # 7. Compute Covariance Matrices for gradRho_l terms
    # Done in gradRhoL for now
    # 8. Run surface detection (only if free surface)
    #9. Compute gradRho and gradRhoL
    if schemeConfig.diffusionParams.densityDiffusionTerm == DensityDiffusionScheme.denormalized or schemeConfig.diffusionParams.densityDiffusionTerm == DensityDiffusionScheme.denormalizedOnly:
        gradRho = computeGradRho(currentState, config, schemeConfig, adjacency)
    else:
        gradRho = None

    if schemeConfig.diffusionParams.densityDiffusionTerm == DensityDiffusionScheme.deltaSPH or schemeConfig.diffusionParams.densityDiffusionTerm == DensityDiffusionScheme.deltaOnly:
        gradRhoL = computeGradRhoL(currentState, config, schemeConfig, adjacency, None)
    else:
        gradRhoL = None


    # 10. Compute drhodt_diss
    drhodt_diss = computeDensityDiffusion(currentState, config, schemeConfig, adjacency, gradRho, gradRhoL)


    # 11. Compute dvdt_diss
    # schemeConfig.diffusionParams.C_l = 0.01 # alpha parameter for delta-SPH
    # schemeConfig.diffusionParams.C_q = 0.0
    # schemeConfig.diffusionParams.viscosityTerm = ViscosityTerms.Monaghan1992.value


    # Use one explicit kinematic viscosity for both integration and analysis.
    # nu_visc_local = extraData.get('nu_visc', 1e-3)

    dvdt_diss = computeVelocityDiffusion(currentState, config, schemeConfig, adjacency)
    
    # 12. Compute drhodt
    drhodt = computeMomentum(currentState, config, schemeConfig, adjacency)

    # 13. Compute dvdt from pressure
    dvdt = computePressureForceSurfaceAware(currentState, config, schemeConfig, adjacency)

    # 14. Apply forcing
    forcing = computeForcing(currentSystem, config.dt, currentSystem.t, config, schemeConfig)
    dvdt += forcing / currentState.masses.view(-1,1)

    # 15. build update
    update = WeaklyCompressibleSystemUpdate(
        dxdt = currentState.velocities.clone(),
        dvdt = dvdt + dvdt_diss,
        drhodt = drhodt + drhodt_diss,
        passive = torch.zeros(currentState.densities.shape, device=currentState.densities.device, dtype=torch.bool)
    )

    # 16. Enforce BCs on update
    enforceUpdates(update, currentSystem, config.dt, currentSystem.t, config, schemeConfig)

    return update, adjacency, currentState