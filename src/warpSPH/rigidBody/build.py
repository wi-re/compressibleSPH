
import torch
from ..configurations.weaklyCompressible import RegionType, ParticleRegion, RigidBody

def buildRigidBody(particleState, regions, bodyId):
    particleIndices = torch.logical_and(particleState.kinds == 1, particleState.materials == bodyId)
    ghostIndices = torch.logical_and(particleState.kinds == 2, particleState.materials == bodyId)

    boundaryRegions = [region for region in regions if region.type == RegionType.Boundary]
    currentRegion = boundaryRegions[bodyId] if len(boundaryRegions) > bodyId else None

    # print(torch.sum(particleIndices), torch.sum(ghostIndices))
    if(torch.sum(particleIndices) == 0):
        print('No particles in body', bodyId)
        return None

    masses = particleState.masses[particleIndices]
    positions = particleState.positions[particleIndices]
    device = particleState.positions.device
    dtype = particleState.positions.dtype

    ghostPositions = particleState.positions[ghostIndices]

    mass = torch.sum(masses)
    centerOfMass = torch.sum(masses.view(-1,1) * positions, dim = 0) / mass
    angularVelocity = torch.tensor(0.0, device = device, dtype = dtype)
    linearVelocity = torch.tensor([0.0, 0.0], device = device, dtype = dtype)
    inertia = torch.sum(masses * torch.linalg.norm(positions - centerOfMass, dim = 1)**2)
    orientation = torch.tensor(0.0, device = device, dtype = dtype)

    return RigidBody(
        centerOfMass=centerOfMass,
        orientation=orientation,
        angularVelocity=angularVelocity,
        linearVelocity=linearVelocity,
        mass=mass,
        inertia=inertia,

        particlePositions=positions - centerOfMass,
        ghostParticlePositions=ghostPositions - centerOfMass,
        particleVelocities=particleState.velocities[particleIndices],


        particleMasses = masses,
        particleUIDs = particleState.UIDs[particleIndices],
        ghostParticleUIDs = particleState.UIDs[ghostIndices],
        particleIndices = particleIndices,
        ghostParticleIndices=ghostIndices,

        particleBoundaryDistances=torch.linalg.norm(particleState.ghostOffsets[particleIndices], dim = -1),
        ghostParticleBoundaryDistances=torch.linalg.norm(particleState.ghostOffsets[ghostIndices], dim = -1),
        particleBoundaryNormals=particleState.ghostOffsets[particleIndices],
        ghostParticleBoundaryNormals=particleState.ghostOffsets[ghostIndices],

        bodyID=bodyId,
        sdf = currentRegion.sdf,
        kind= currentRegion.kind
    )
