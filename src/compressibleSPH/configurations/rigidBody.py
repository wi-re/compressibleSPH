from .moduleConfigurations.boundaryConditions import BoundaryCondition, BoundaryConditionType, boundaryConditionToDict, dictToBoundaryCondition, BCType
import torch
from dataclasses import dataclass, field

@dataclass(slots = True)
class RigidBody:
    centerOfMass: torch.Tensor
    orientation: torch.Tensor
    angularVelocity: torch.Tensor
    linearVelocity: torch.Tensor
    mass: torch.Tensor
    inertia: torch.Tensor
    
    # All in world coordinates
    particlePositions: torch.Tensor
    particleVelocities: torch.Tensor
    particleMasses: torch.Tensor
    particleUIDs: torch.Tensor
    particleIndices: torch.Tensor
    particleBoundaryDistances: torch.Tensor
    particleBoundaryNormals: torch.Tensor
    
    ghostParticlePositions: torch.Tensor
    ghostParticleIndices: torch.Tensor  
    ghostParticleUIDs: torch.Tensor
    ghostParticleBoundaryDistances: torch.Tensor
    ghostParticleBoundaryNormals: torch.Tensor
    
    
    sdf: callable
    bodyID: int = 0  
    kind: BCType = BCType.constant