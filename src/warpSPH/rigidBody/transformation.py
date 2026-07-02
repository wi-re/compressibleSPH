import torch

from ..configurations.rigidBody import RigidBody

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
