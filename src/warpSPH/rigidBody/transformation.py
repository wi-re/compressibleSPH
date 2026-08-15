"""Builds a rigid body's world-space transform -- rotation about the origin
then translation to `centerOfMass` -- as a 3x3 homogeneous matrix, so
`update.updateBodyParticlesWCSPH` can move the body's rest-pose particle/ghost
positions with one matrix multiply. Hardcoded to 2D (a single z-axis rotation
angle), matching the rest of the mDBC rigid-body path.
"""

import torch

from ..configurations.rigidBody import RigidBody

__all__ = ['getTransformationMatrix']


def getTransformationMatrix(rigidBody : RigidBody):
    device = rigidBody.centerOfMass.device
    dtype = rigidBody.centerOfMass.dtype
    I = torch.eye(3, device = device, dtype = dtype)
    R = torch.tensor([[torch.cos(rigidBody.orientation), -torch.sin(rigidBody.orientation), 0],
                      [torch.sin(rigidBody.orientation), torch.cos(rigidBody.orientation), 0],
                      [0, 0, 1]], device = device, dtype = dtype)
    T = torch.tensor([[1, 0, rigidBody.centerOfMass[0]],
                      [0, 1, rigidBody.centerOfMass[1]],
                      [0, 0, 1]], device = device, dtype = dtype)
    return T @ R @ I
