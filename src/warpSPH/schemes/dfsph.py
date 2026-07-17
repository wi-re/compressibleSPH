from warpSPH.configurations import *
from sphWarpCore.enumTypes import ViscosityTerms
from sphWarpCore.kernels.wp_kernel import sphKernel_xi
from warpSPH.systems import *
from warpSPH.modules import *
from warpSPH.enumTypes import *
from sphWarpCore import *

from warpSPH.utils.timer import TimedBlock
from torch.profiler import profile, record_function, ProfilerActivity

from ..systems.incompressible import IncompressibleSystem, IncompressibleState, IncompressibleSystemUpdate

import numpy as np
def dfsph_step(
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
    # with TimedBlock('compute adjacency', use_cuda=True, device=config.device) as tb_adjacency:
    with record_function("[warpSPH] - [deltaSPH - 01] - compute adjacency"):
        verletScale = config.verletScale

        adjacency = buildVerletList(
            currentState, 
            config.domain, verletScale = verletScale, supportMode = SupportScheme.SuperSymmetric,
            priorNeighborhood = currentSystem.adjacency,
            verbose = False)
        currentSystem.adjacency = adjacency

    # 2. Compute density if density is none
    # with TimedBlock('compute density', use_cuda=True, device=config.device) as tb_density:
    with record_function("[warpSPH] - [deltaSPH - 02] - compute density"):
        if currentState.densities is None or not schemeConfig.solverConfig.integrateRho:
            # print("Computing densities...")
            currentState.densities = computeDensities(currentState, config, schemeConfig, adjacency)

    # print(f'step {currentSystem.t:.6g}: density stats: mean={currentState.densities.mean().item():.6g}, min={currentState.densities.min().item():.6g}, max={currentState.densities.max().item():.6g}')
    # if currentState.densities.max() > 1.01:
        # print(f"Warning: Density exceeds 1.01, max density: {currentState.densities.max().item():.6g}")
    # 3. Skipped mDBC density computation since no boundaries are present
    # with TimedBlock('compute mDBC density', use_cuda=True, device=config.device) as tb_mdbc:
    with record_function("[warpSPH] - [deltaSPH - 03] - compute mDBC density"):
        currentState.densities = computeMdbcDensity(currentState, config, schemeConfig, adjacency)

        # print(f'Fluid density stats: min={currentState.densities[currentState.kinds == 0].min().item()}, max={currentState.densities[currentState.kinds == 0].max().item()}, mean={currentState.densities[currentState.kinds == 0].mean().item()}')
        # print(f'Boundary density stats: min={currentState.densities[currentState.kinds == 1].min().item()}, max={currentState.densities[currentState.kinds == 1].max().item()}, mean={currentState.densities[currentState.kinds == 1].mean().item()}')
    # 4. enforce BCs
    with record_function("[warpSPH] - [deltaSPH - 06] - compute boundary velocities"):
        currentVelocities = currentState.velocities.clone()
        currentState.velocities = computeBoundaryVelocities(currentState, config, schemeConfig, adjacency)
    

    # with TimedBlock('enforce BCs', use_cuda=True, device=config.device) as tb_bcs:
    with record_function("[warpSPH] - [deltaSPH - 04] - enforce BCs"):
        enforceDirichlet(currentSystem, currentSystem.t, config.dt, config, schemeConfig)
    # 5. compute EOS (WC version)
    # with TimedBlock('compute EOS', use_cuda=True, device=config.device) as tb_eos:
    # with record_function("[warpSPH] - [deltaSPH - 05] - compute EOS"):
        # currentState.pressures = weaklyCompressibleEOS(currentState, schemeConfig)
    # 6. Skipped boundary velocity computation since no boundaries are present
    # with TimedBlock('compute boundary velocities', use_cuda=True, device=config.device) as tb_boundary_velocities:

    # 7. Compute Covariance Matrices for gradRho_l terms
    # Done in gradRhoL for now
    # 8. Run surface detection (only if free surface)
    # with TimedBlock('compute surface detection', use_cuda=True, device=config.device) as tb_surface:
    with record_function("[warpSPH] - [deltaSPH - 08] - compute surface detection"):
        fs, fsm, n, renormalizationState_, lMin = detectFreeSurface(currentState, config, schemeConfig, schemeConfig.surfaceDetectionConfig, adjacency, returnNormals = True) 
        currentState.surfaceIndicators = (fsm > 0.5).to(torch.int32)
        currentState.surfaceNormals = n
        currentState.surfaceLambdas = lMin

    # #9. Compute gradRho and gradRhoL
    # # with TimedBlock('compute gradRho', use_cuda=True, device=config.device) as tb_gradRho:
    # with record_function("[warpSPH] - [deltaSPH - 09] - compute gradRho and gradRhoL"):
    #     if schemeConfig.diffusionParams.densityDiffusionTerm == DensityDiffusionScheme.denormalized or schemeConfig.diffusionParams.densityDiffusionTerm == DensityDiffusionScheme.denormalizedOnly:
    #         gradRho = computeGradRho(currentState, config, schemeConfig, adjacency)
    #     else:
    #         gradRho = None
    # # with TimedBlock('compute gradRhoL', use_cuda=True, device=config.device) as tb_gradRhoL:
    # with record_function("[warpSPH] - [deltaSPH - 09] - compute gradRhoL"):
    #     if schemeConfig.diffusionParams.densityDiffusionTerm == DensityDiffusionScheme.deltaSPH or schemeConfig.diffusionParams.densityDiffusionTerm == DensityDiffusionScheme.deltaOnly:
    #         gradRhoL = computeGradRhoL(currentState, config, schemeConfig, adjacency, L = renormalizationState_)
    #     else:
    #         gradRhoL = None


    # # 10. Compute drhodt_diss
    # # with TimedBlock('compute drhodt_diss', use_cuda=True, device=config.device) as tb_drhodt_diss:
    # with record_function("[warpSPH] - [deltaSPH - 10] - compute drhodt_diss"):
    #     drhodt_diss = computeDensityDiffusion(currentState, config, schemeConfig, adjacency, gradRho, gradRhoL)

    # 11. Compute dvdt_diss
    # with TimedBlock('compute dvdt_diss', use_cuda=True, device=config.device) as tb_dvdt_diss:
    with record_function("[warpSPH] - [deltaSPH - 11] - compute dvdt_diss"):
        dvdt_diss = computeVelocityDiffusion(currentState, config, schemeConfig, adjacency)
    
    # 12. Compute drhodt
    # with TimedBlock('compute drhodt', use_cuda=True, device=config.device) as tb_drhodt:
    with record_function("[warpSPH] - [deltaSPH - 12] - compute drhodt"):
        drhodt = computeMomentum(currentState, config, schemeConfig, adjacency)

    # 14. Apply forcing
    # with TimedBlock('compute forcing', use_cuda=True, device=config.device) as tb_forcing:
    with record_function("[warpSPH] - [deltaSPH - 14] - compute forcing"):
        forcing = computeForcing(currentSystem, config.dt, currentSystem.t, config, schemeConfig)
        dvdt = forcing / currentState.masses.view(-1,1)


    # 15. Compute gravity
    # with TimedBlock('compute gravity', use_cuda=True, device=config.device) as tb_gravity:
    with record_function("[warpSPH] - [deltaSPH - 15] - compute gravity"):
        gravity = computeGravity(currentState, config, schemeConfig, adjacency)
        dvdt += gravity

    # Revert boundary velocity
    # with TimedBlock('compute mDBC no-pen shift', use_cuda=True, device=config.device) as tb_nopenshift:
    with record_function("[warpSPH] - [deltaSPH - 16] - compute mDBC no-pen shift"):
        nopenshift = computeMdbcNoPenShift(currentState, config, schemeConfig, adjacency)
        dvdt += nopenshift / dt
    # currentState.velocities = currentVelocities

    # First we project the velocity field to be divergence free using the DFSph solver
    # This is the standard incompressible approach
    # However, this can lead to issues with particle disorder and clustering 
    currentState.pressures = torch.zeros_like(currentState.densities) if currentState.pressures is None else currentState.pressures
    dvdt_pressure, pressure, errors, pressures = solveDivergenceFree(
        particles = currentState,
        config = config,
        schemeConfig = schemeConfig,
        adjacency = adjacency,
        dvdt = dvdt + dvdt_diss,
        dt = dt
    )
    currentState.pressures = pressure

    # To resolve the issues with particle disorder and clustering, we can also solve for the incompressible pressure using the Incompressible SPH solver
    # This effectively acts as a particle shifting term that helps to maintain particle order and prevent clustering
    # Instead of being explicit, e.g., as in delta SPH, this is an implicit particle shift
    # dvdt_incomp, pressure_incomp, errors_incomp, pressures_incomp = solveIncompressible(
    #     particles = currentState,
    #     config = config,
    #     schemeConfig = schemeConfig,
    #     adjacency = adjacency,
    #     dvdt = dvdt + dvdt_diss + dvdt_pressure,
    #     dt = dt
    # )
    # The shift is directly applied to the particle positions
    # As the timestep for our tests is small the shift is small and does not significantly affect the velocity field
    vPrime = currentState.velocities + dt * (dvdt + dvdt_diss + dvdt_pressure)

    # gradV = -warpOperation(
    #     queryParticles=currentState,
    #     operationProperties=OperationProperties(
    #         kernel = config.kernel,
    #         operation = WarpOperation.Gradient,
    #         gradientMode = GradientScheme.Difference,
    #         supportMode = SupportScheme.Scatter,
    #     ),
    #     queryValues = vPrime,
    #     domain = config.domain,
    #     adjacency=adjacency,
    # )
    # vCorrection = torch.einsum('nij, ni->nj', gradV, dt * dvdt_incomp)
    # dvdt_correction = vCorrection / dt
    # dvdt += dvdt_correction


    # 16. build update
    # with TimedBlock('build update', use_cuda=True, device=config.device) as tb_update:
    with record_function("[warpSPH] - [deltaSPH - 16] - build update"):
        update = WeaklyCompressibleSystemUpdate(
            dxdt = currentState.velocities.clone(),# + dt * dvdt_incomp,
            dvdt = dvdt + dvdt_diss + dvdt_pressure,# + dvdt_incomp,
            drhodt = drhodt,# + drhodt_diss,
            passive = torch.zeros(currentState.densities.shape, device=currentState.densities.device, dtype=torch.bool)
        )
    # update.drhodt = update.drhodt

    # 17. Enforce BCs on update
    # with TimedBlock('enforce updates', use_cuda=True, device=config.device) as tb_enforce:
    with record_function("[warpSPH] - [deltaSPH - 17] - enforce updates"):
        enforceUpdates(update, currentSystem, config.dt, currentSystem.t, config, schemeConfig)
        update.dxdt[currentState.kinds != 0,:] = 0.0
        update.dvdt[currentState.kinds != 0,:] = 0.0
        update.drhodt[currentState.kinds != 0] = 0.0

    # performanceDict = {
    #     'tb_adjacency': tb_adjacency,
    #     'tb_density': tb_density,
    #     'tb_mdbc': tb_mdbc,
    #     'tb_bcs': tb_bcs,
    #     'tb_eos': tb_eos,
    #     'tb_boundary_velocities': tb_boundary_velocities,
    #     'tb_surface': tb_surface,
    #     'tb_gradRho': tb_gradRho,
    #     'tb_gradRhoL': tb_gradRhoL,
    #     'tb_drhodt_diss': tb_drhodt_diss,
    #     'tb_dvdt_diss': tb_dvdt_diss,
    #     'tb_drhodt': tb_drhodt,
    #     'tb_dvdt': tb_dvdt,
    #     'tb_forcing': tb_forcing,
    #     'tb_gravity': tb_gravity,
    #     'tb_nopenshift': tb_nopenshift,
    #     'tb_update': tb_update,
    #     'tb_enforce': tb_enforce,
    # }

    # for key, tb in performanceDict.items():
    #     tb.cuda_ms = tb._start_event.elapsed_time(tb._end_event) if tb.use_cuda else None
    #     print(f"[{key}] CPU: {tb.cpu_ms:.3f} ms | CUDA: {tb.cuda_ms:.3f} ms, ratio {tb.cuda_ms / tb.cpu_ms if tb.cuda_ms is not None else 'N/A'}")

    return update, adjacency, currentState, (errors,pressures)#, (errors_incomp, pressures_incomp)