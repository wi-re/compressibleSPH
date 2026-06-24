from .transformation import getTransformationMatrix
from ..systems.weaklyCompressible import WeaklyCompressibleState
from ..configurations.weaklyCompressible import RegionType, ParticleRegion, RigidBody, BoundaryConditionType
import torch

def updateBodyParticlesWCSPH(particleState, rigidBody: RigidBody):
    T = getTransformationMatrix(rigidBody)
    # print(T)

    particlePositions = torch.einsum('uv, nv -> nu', T, torch.hstack([
        rigidBody.particlePositions, 
        torch.ones_like(rigidBody.particlePositions[:,0]).view(-1,1)]))[:,:2]
    ghostParticlePositions  = torch.einsum('uv, nv -> nu', T, torch.hstack([
        rigidBody.ghostParticlePositions, 
        torch.ones_like(rigidBody.ghostParticlePositions[:,0]).view(-1,1)]))[:,:2]
    
    offsets = particlePositions - ghostParticlePositions
    relativePositions = particlePositions - rigidBody.centerOfMass
    
    # print(rigidBody.angularVelocity)
    # print(torch.stack([relativePositions[:,1], -relativePositions[:,0]], dim = 1))

    particleVelocities = torch.stack([-relativePositions[:,1], relativePositions[:,0]], dim = 1) * rigidBody.angularVelocity + rigidBody.linearVelocity


    # particleVelocities += rigidBody.linearVelocity

    updatedPositions = particleState.positions.clone()
    updatedPositions[rigidBody.particleIndices] = particlePositions
    updatedPositions[rigidBody.ghostParticleIndices] = ghostParticlePositions

    updatedVelocities = particleState.velocities.clone()
    if rigidBody.kind != BoundaryConditionType.constant:
        # print('updating velocities')
        # print(particleVelocities)
        updatedVelocities[rigidBody.particleIndices] = particleVelocities
        updatedVelocities[rigidBody.ghostParticleIndices] = particleVelocities

    updatedOffsets = particleState.ghostOffsets.clone()
    updatedOffsets[rigidBody.particleIndices] = offsets
    updatedOffsets[rigidBody.ghostParticleIndices] = offsets

    return WeaklyCompressibleState(
        positions = updatedPositions,
        supports = particleState.supports,
        masses = particleState.masses,
        densities = particleState.densities,
        velocities=updatedVelocities,

        pressures = particleState.pressures,
        soundspeeds=particleState.soundspeeds,

        kinds = particleState.kinds,
        materials = particleState.materials,
        UIDs = particleState.UIDs,

        UIDcounter=particleState.UIDcounter,

        ghostIndices = particleState.ghostIndices,
        ghostOffsets = updatedOffsets,

    )
