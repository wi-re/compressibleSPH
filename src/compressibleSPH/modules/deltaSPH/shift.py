# 17. If finalize, compute shifting and update positions and velocities
# from sphWarpCore.radius import HashMapLengthMode

from sphWarpCore import KernelFunctions, KernelFunctions, OperationProperties, SupportScheme, WarpOperation, buildVerletList

from sphWarpCore.enumTypes import HashMapLengthMode, SupportScheme, WarpOperation
from sphWarpCore import warpOperation

from compressibleSPH.sample.wp_deltaShift import computeDeltaShiftWarp

def computeDeltaShift(currentState, config, schemeConfig, domain, adjacency):
    # for i in tqdm(range(shiftIters), leave = False):
    original_positions = currentState.positions.clone()
    original_densities = currentState.densities.clone()
    for i in range(schemeConfig.shiftProperties.iterations):
            
        adjacency = buildVerletList(
            currentState, 
            config.domain, verletScale = config.verletScale, supportMode = SupportScheme.SuperSymmetric,
            priorNeighborhood = adjacency,
            verbose = False)


        currentState.densities =warpOperation(
            currentState,
            operationProperties = OperationProperties(
                operation=WarpOperation.Density,
                kernel = KernelFunctions.Wendland6, 
                supportMode = SupportScheme.Gather
            ),
            domain = domain,
            adjacency = adjacency
        )
        # display(currentState)


        shift = computeDeltaShiftWarp(
            currentState,
            operationProperties = OperationProperties(
                operation=WarpOperation.Density,
                kernel = KernelFunctions.Wendland6, 
                supportMode = SupportScheme.Gather
            ),
            referenceParticles = currentState,
            domain = domain,
            # supportMode = SupportScheme.Gather,
            # kernel = KernelFunctions.Wendland2,
            # operationMode = OperationDirection.AllToAll,
            adjacency = adjacency,

            CFL = schemeConfig.shiftProperties.CFL, computeMach = schemeConfig.shiftProperties.computeMach, c_max = schemeConfig.shiftProperties.maxC,
            rho0 = 1.0, dx = config.dx.item(),
        )

        # print(f'Iteration {i}, shift magnitude: {shift.norm(dim=1).mean().item()}')
        currentState.positions = currentState.positions + shift

    delta = currentState.positions - original_positions
    currentState.positions = original_positions
    currentState.densities = original_densities

    return delta, adjacency