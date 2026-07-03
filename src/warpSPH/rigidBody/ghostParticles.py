import torch
# from ..systems.weaklyCompressible import WeaklyCompressibleState
from ..configurations.weaklyCompressible import RegionType, ParticleRegion
from typing import Any, Optional, Union

def addBoundaryGhostParticles(regions, particleState : Any):
    device = particleState.positions.device
    dtype = particleState.positions.dtype
    
    if not torch.any(particleState.kinds == 1):
        # print("No Boundary particles found. Returning original state.")
        return particleState
    
    ghostIndices = particleState.positions.new_ones(particleState.positions.shape[0], dtype = torch.int64) * -1
    ghostOffsets = torch.zeros_like(particleState.positions)
    
    boundaryMaterial = 0
    numParticles = particleState.positions.shape[0]
    ghostPositions = []
    boundaryIndices = []
    
    for region in regions:
        # print(f"Processing region {region} of type {region.type}")
        if region.type == RegionType.Boundary:
            particleIndices = torch.arange(particleState.positions.shape[0], device = device, dtype = torch.int64)
            relevantParticles = torch.logical_and(particleState.kinds == 1, particleState.materials == boundaryMaterial)
            # print('Boundary region', boundaryMaterial, 'has', torch.sum(relevantParticles).item(), 'fluid particles.')
            relevantParticles = particleIndices[relevantParticles]
            
            sdfValues, sdfNormals = region.sdf(particleState.positions[relevantParticles])
            clampedDist = sdfValues - torch.min(-sdfValues, particleState.supports.mean())
            offsets = clampedDist.view(-1,1) * sdfNormals
            
            bIndices = relevantParticles
            boundaryIndices.append(bIndices)
            gUIDs = particleState.UIDs[relevantParticles]
            gIndices = torch.arange(numParticles, numParticles + offsets.shape[0], device = device, dtype = torch.int64)
            ghostIndices[bIndices] = gIndices
            ghostOffsets[bIndices] = offsets
            # print(bIndices, gIndices)

            # ghostIndices[gIndices] = bIndices
            # ghostOffsets[gIndices] = -offsets
            numParticles += offsets.shape[0]
            # print(sdfValues)
            
            
            ghostPositions.append(particleState.positions[relevantParticles] - offsets)
            boundaryMaterial += 1

            # print(f"Added {offsets.shape[0]} ghost particles for boundary region {region}.")
            
    boundaryIndices = torch.cat(boundaryIndices, dim = 0)
    WeaklyCompressibleState = type(particleState)
    return WeaklyCompressibleState(
        positions = torch.cat([particleState.positions, torch.cat(ghostPositions, dim = 0)], dim = 0),
        supports = torch.cat([particleState.supports, particleState.supports[boundaryIndices]], dim = 0),
        masses = torch.cat([particleState.masses, particleState.masses[boundaryIndices]], dim = 0),
        densities = torch.cat([particleState.densities, particleState.densities[boundaryIndices]], dim = 0),
        velocities = torch.cat([particleState.velocities, particleState.velocities[boundaryIndices]], dim = 0),
        
        pressures= torch.cat([particleState.pressures, particleState.pressures[boundaryIndices]], dim = 0) if particleState.pressures is not None else None,
        soundspeeds= torch.cat([particleState.soundspeeds, particleState.soundspeeds[boundaryIndices]], dim = 0) if particleState.soundspeeds is not None else None,
        
        kinds = torch.cat([particleState.kinds, torch.ones_like(boundaryIndices).to(particleState.kinds.dtype) * 2], dim = 0),
        materials = torch.cat([particleState.materials, particleState.materials[boundaryIndices]], dim = 0),
        UIDs = torch.cat([particleState.UIDs, -particleState.UIDs[boundaryIndices]], dim = 0),
        
        ghostIndices = torch.cat([ghostIndices, boundaryIndices], dim = 0),
        ghostOffsets= torch.cat([ghostOffsets, -ghostOffsets[boundaryIndices]], dim = 0),
        
        UIDcounter = particleState.UIDcounter
    )
            
