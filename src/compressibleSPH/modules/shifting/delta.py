# 17. If finalize, compute shifting and update positions and velocities
# from sphWarpCore.radius import HashMapLengthMode

from sphWarpCore import KernelFunctions, KernelFunctions, OperationProperties, SupportScheme, WarpOperation, buildVerletList

from sphWarpCore.enumTypes import HashMapLengthMode, SupportScheme, WarpOperation
from sphWarpCore import warpOperation
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
        )

        # print(f'Iteration {i}, shift magnitude: {shift.norm(dim=1).mean().item()}')
        currentState.positions = currentState.positions + shift

    delta = currentState.positions - original_positions
    currentState.positions = original_positions
    currentState.densities = original_densities

    return delta, adjacency