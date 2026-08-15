"""Assemble a ``WaveSystemv3`` from the grids the case generator produced.

Takes the per-particle ``u``/``v``/``c``/``damping`` grids plus the two id
grids from :func:`warpSPH.caseUtils.waveEquation.gencase.genInitial`, resolves
source ids to amplitudes and obstacle/boundary ids to wave speeds, optionally
smooths the initial conditions, and returns the system together with its
timestep.

This is the last stage of the wave pipeline, not an entry point -- there is no
``Case`` wrapping it, so a caller has to run the earlier stages themselves.
"""

import torch
from ..systems.waveSystem import computeDt, WaveSystemv3, WaveSystemStatev3
from ..caseUtils.waveEquation.sample import smoothValuesWarp
from warpSPHCore import *

def finalizeWaveSystemSetup(
    particleState,
    uGrid, vGrid, cGrid, dampGrid,
    uSourceGrid, cSourceGrid,
    sourceMagnitudes, obstacleSpeeds,
    config, caseConfig,
):


    boundaryIds = torch.unique(cSourceGrid)
    boundaryIds = boundaryIds[boundaryIds != 0]  # Exclude background (0)
    # print("Boundary IDs:", boundaryIds)

    cGrid = torch.full_like(cGrid, caseConfig.defaultSpeed)
    # print(torch.sum(cGrid), '/', cGrid.numel(), ' default: ', caseConfig.defaultSpeed)
    for bId in boundaryIds:
        if bId == -1:
            speed = caseConfig.defaultBoundarySpeed
        else:
            speed = obstacleSpeeds[bId-1]
        cGrid = torch.where(cSourceGrid == bId, torch.full_like(cGrid, speed), cGrid)
        # print(f"Set speed {speed} for boundary ID {bId}")
    # print(torch.sum(cGrid))

    uGrid = torch.zeros_like(uGrid)
    sourceIds = torch.unique(uSourceGrid)
    sourceIds = sourceIds[sourceIds != 0]  # Exclude background (0)
    for sId in sourceIds:
        magnitude = sourceMagnitudes[sId-1]  # source IDs are 1-indexed in uSourceGrid
        uGrid = torch.where(uSourceGrid == sId, torch.full_like(uGrid, magnitude), uGrid)
        # print(f"Set magnitude {magnitude} for source ID {sId}")

        
    if caseConfig.smoothICs:
        # if args.verbose:
            # print("Smoothing initial conditions...")
        uGrid = smoothValuesWarp(
            uGrid,
            particleState,
            caseConfig.smoothIterations, None,
            config
        )
    # uGrid[particleState.positions[:,0] > 0] = 0.0

        
    waveState = WaveSystemStatev3(
        positions=particleState.positions,
        supports=particleState.supports,
        masses=particleState.masses,
        densities=particleState.densities,
        
        kinds=torch.zeros(particleState.positions.shape[0], device=config.device, dtype=torch.int32),
        materials=torch.zeros(particleState.positions.shape[0], device=config.device, dtype=torch.int32),
        UIDs=torch.arange(particleState.positions.shape[0], device=config.device, dtype=torch.int32),
        UIDcounter=particleState.positions.shape[0],
        
        u=uGrid,
        v=vGrid,
        c=cGrid,
        damping=dampGrid
    )
    adjacency = buildVerletList(waveState, config.domain, 1.0, SupportScheme.SuperSymmetric, None)


    waveSystem = WaveSystemv3(
        state = waveState,
        adjacency = adjacency,
        domain = config.domain,
        t = torch.tensor(0.0, device=config.device, dtype=config.dtype)
    )

    # print(obstacleSpeeds + [caseConfig.defaultSpeed])
    dt = config.dt if not config.adaptiveDt else computeDt(waveSystem, config, caseConfig, None, obstacleSpeeds, False)
    # config.dt = dt

    return waveSystem, dt