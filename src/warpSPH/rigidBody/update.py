"""Applies a `RigidBody`'s current transform to its owning particle state:
moves the body's own and its ghost particles' positions via
`transformation.getTransformationMatrix`, derives their rigid-body velocity
from angular velocity about the center of mass (only written into
`particleState.velocities` when the body's `kind` is `BCType.constant`), and
rewrites the boundary/ghost offset pair to match the new geometry. Called
every step from `systems/weaklyCompressible.py`/`systems/incompressible.py`'s
`finalize`, right after `rigidBody.integrate.integrateRigidBody` advances the
pose. Rebuilds the state via `type(particleState)` with the same keyword set
`WeaklyCompressibleState` and `IncompressibleState` both declare, rather than
mutating in place, so it works for either.
"""

from .transformation import getTransformationMatrix
# from ..systems.weaklyCompressible import WeaklyCompressibleState
from ..configurations.weaklyCompressible import RegionType, ParticleRegion, RigidBody, BoundaryConditionType, BCType
import torch
from typing import Any, Optional, Union

__all__ = ['updateBodyParticlesWCSPH']


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
    if rigidBody.kind == BCType.constant:
        # print('updating velocities')
        # print(particleVelocities)
        updatedVelocities[rigidBody.particleIndices] = particleVelocities
        updatedVelocities[rigidBody.ghostParticleIndices] = particleVelocities
        # print(f'updated velocities to {particleVelocities[0]}')

    updatedOffsets = particleState.ghostOffsets.clone()
    updatedOffsets[rigidBody.particleIndices] = offsets
    updatedOffsets[rigidBody.ghostParticleIndices] = offsets
    WeaklyCompressibleState = type(particleState)
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

        surfaceIndicators = particleState.surfaceIndicators,
        surfaceNormals = particleState.surfaceNormals,
        surfaceLambdas = particleState.surfaceLambdas,

        

    )
