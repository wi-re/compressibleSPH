# 17. If finalize, compute shifting and update positions and velocities
# from sphWarpCore.radius import HashMapLengthMode

from sphWarpCore import KernelFunctions, KernelFunctions, OperationProperties, SupportScheme, WarpOperation, buildVerletList

from sphWarpCore.enumTypes import HashMapLengthMode, SupportScheme, WarpOperation
from sphWarpCore import warpOperation
from sphWarpCore.kernels.wp_kernel import sphKernel_xi, sphKernelScale
import torch

from compressibleSPH.sample.wp_deltaShift import computeDeltaShiftWarp

    # for i in tqdm(range(shiftIters), leave = False):
def computeDeltaShift(currentState, config, schemeConfig, domain, adjacency, iters = -1):
    original_positions = currentState.positions.clone()
    original_densities = currentState.densities.clone()
    for i in range(schemeConfig.shiftProperties.iterations if iters == -1 else iters):
            
        adjacency = buildVerletList(
            currentState, 
            config.domain, verletScale = config.verletScale, supportMode = SupportScheme.SuperSymmetric,
            priorNeighborhood = adjacency,
            verbose = False)


        # currentState.densities =warpOperation(
        #     currentState,
        #     operationProperties = OperationProperties(
        #         operation=WarpOperation.Density,
        #         kernel = config.kernel, 
        #         supportMode = SupportScheme.Gather
        #     ),
        #     domain = domain,
        #     adjacency = adjacency
        # )
        # display(currentState)

        velocity_magnitudes = torch.linalg.vector_norm(currentState.velocities, dim=-1)
        finite_velocity_magnitudes = velocity_magnitudes[torch.isfinite(velocity_magnitudes)]
        v_max = (
            torch.max(finite_velocity_magnitudes)
            if finite_velocity_magnitudes.numel() > 0
            else torch.tensor(float('nan'), device=currentState.velocities.device)
        )
        c_max = v_max / schemeConfig.fluid.fixedSoundSpeed
        h_min = currentState.supports.min()
        # print(f'Iteration {i}, max velocity: {v_max.item()}, min support: {h_min.item()}, c_max: {c_max.item()}')


        shift = computeDeltaShiftWarp(
            currentState,
            operationProperties = OperationProperties(
                operation=WarpOperation.Density,
                kernel = config.kernel, 
                supportMode = SupportScheme.Gather
            ),
            referenceParticles = currentState,
            domain = domain,
            # supportMode = SupportScheme.Gather,
            # kernel = KernelFunctions.Wendland2,
            # operationMode = OperationDirection.AllToAll,
            adjacency = adjacency,

            CFL = schemeConfig.shiftProperties.CFL, computeMach = schemeConfig.shiftProperties.computeMach, c_max = c_max.cpu().item(),
            rho0 = 1.0, dx = config.dx.item(),
        ) #* schemeConfig.fluid.fixedSoundSpeed * config.dt
        # The compute function returns the unscaled term
        # \sum_j m_j * [ 2 / (rho_i + rho_j) ] * [ 1 + R * (w_ij / W_0)^n ] * gradW_ij
        # The scaling factor is applied here to get the final shift amount
        Ma = c_max
        c0 = schemeConfig.fluid.fixedSoundSpeed
        CFL = schemeConfig.shiftProperties.CFL
        kernelScale = float(sphKernelScale(config.kernel.value, config.dim))
        h = currentState.supports / kernelScale
        dt = config.dt

        # If we follow the delta^+ approach we get the scaling
        # - CFL * Ma * 2 h^2 (note that we include the 2 from the mean density term in the computation of the shift so its not (2h)^2 as in (7) in Sun et al. 2017)
        scalingDeltaPlus = -CFL * Ma * 2 * h**2

        # If we follow the approach of Michel 2022 we can rewrite the shift as a shifting velocity instead of a shift amount, and then scale it by dt to get the final shift amount
        # The scaling factor here is
        # - Ma * c0 * 2 h
        # Note that the acoustic time step is dt = CFL * h / c0, so the scaling factor is equivalent to the delta^+ scaling factor for a fixed time conservative timestep
        scalingMichel = - Ma * c0 * 2 * h * dt

        dt_c = schemeConfig.shiftProperties.CFL * h.min().cpu().item() / c0# / kernelScale

        # print('-' * 80)
        # print(f'scalingDeltaPlus: {scalingDeltaPlus.mean().item()}, scalingMichel: {scalingMichel.mean().item()}, \n[CFL: {CFL}, Ma: {Ma.item()}, c0: {c0}, h: {h.mean().item()}, dt: {dt}], ratio: {scalingDeltaPlus.mean().item() / scalingMichel.mean().item()}')
        # print(f'dt_c: {dt_c}, dt: {dt}, ratio: {dt_c / dt}')


        shift = shift * scalingDeltaPlus.unsqueeze(-1)

        # print(f'Iteration {i}, shift magnitude: {shift.norm(dim=1).mean().item()}, max: {shift.norm(dim=1).max().item()} [dx: {config.dx.item()}/dt: {config.dt}/mean support: {currentState.supports.mean().item()}]')

        # print(f'Iteration {i}, shift magnitude: {shift.norm(dim=1).mean().item()}, max: {shift.norm(dim=1).max().item()} [dx: {config.dx.item()}/dt: {config.dt}/mean support: {currentState.supports.mean().item()}]')
        currentState.positions = currentState.positions + shift

    delta = currentState.positions - original_positions
    currentState.positions = original_positions
    currentState.densities = original_densities

    return delta, adjacency