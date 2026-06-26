import torch
from .wp_mat import computeLiuMatricesWarp

from sphWarpCore.enumTypes import *
from sphWarpCore import *
from typing import Any
from ...configurations.simulationConfig import SimulationConfig

def interpolateLiuLiu(
    queryPositions: torch.Tensor,
    referenceParticles: Any,
    referenceQuantities: torch.Tensor,
    config: SimulationConfig,
    neighbor_threshold: int = 4,
    direction: OperationDirection = OperationDirection.AllToAll,
):

    _, b, A_g, neighCounts = computeLiuMatricesWarp(
        queryPositions = queryPositions,
        referenceParticles = referenceParticles,
        referenceQuantities = referenceQuantities,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = SupportScheme.Scatter,
            operationMode = direction,
        ),
        domain = config.domain,
    )

    A_g_inv = torch.zeros_like(A_g)
    A_g_inv[neighCounts > neighbor_threshold] = torch.linalg.pinv(A_g[neighCounts > neighbor_threshold])

    res = torch.matmul(A_g_inv, b.unsqueeze(2))[:,:,0]

    return res[:,0], res[:,1:], neighCounts, A_g, b


def liuExtend(
        q: torch.Tensor,
        sim_config: Any,
        scheme_config: Any,
        particles: Any,
        distances: torch.Tensor,
        normals: torch.Tensor,
        current_time: torch.Tensor,
        dt: torch.Tensor,
        neighborThreshold: int = 4,
        defaultValue: float = 0.0,
):
    if len(q.shape) > 1:
        q_flat = q.view(q.shape[0], -1)
        defaults = torch.full((q_flat.shape[1],), defaultValue, device=q.device, dtype=q.dtype) if isinstance(defaultValue, (int, float)) else defaultValue.flatten()
        extended_q = torch.zeros_like(q_flat)
        for i in range(q_flat.shape[1]):
            extended_q[:, i] = liuExtend(
                q = q_flat[:, i],
                sim_config = sim_config,
                scheme_config = scheme_config,
                particles = particles,
                distances = distances,
                normals = normals,
                current_time = current_time,
                dt = dt,
                neighborThreshold = neighborThreshold,
                defaultValue = defaults[i] if defaults.ndim > 0 else defaults.item(),
            )
        return extended_q.view(q.shape)
    


    mask = torch.logical_and(torch.abs(distances) < particles.supports * 2.0, distances < 0)
    masked_positions = particles.positions[mask]
    relPos = 2 * distances[mask].unsqueeze(1) * normals[mask]
    queryPositions = masked_positions - relPos

    shep, b, A_g, neighCounts = computeLiuMatricesWarp(
        queryPositions = queryPositions,
        referenceParticles = particles,
        referenceQuantities = q,
        operationProperties = OperationProperties(
                kernel = sim_config.kernel,
                supportMode = SupportScheme.Scatter,
                operationMode = OperationDirection.FluidToBoundary,
        ),
        domain = sim_config.domain,
    )
    # print(shep)

    A_g_inv = torch.zeros_like(A_g)
    A_g_inv[neighCounts > 4] = torch.linalg.pinv(A_g[neighCounts > 4])

    res = torch.matmul(A_g_inv, b.unsqueeze(2))[:,:,0]

    extended_q = torch.zeros_like(masked_positions[:,0])

    b_scalar = b[:,0]
    b_grad = b[:,1:]

    res_scalar = res[:,0]
    res_grad = res[:,1:]

    # print(res_scalar, res_grad)

    # directions = torch.nn.functional.normalize(normals[mask], dim=1)
    dot = torch.einsum('ij,ij->i', relPos, res_grad)
    projected_q = res_scalar + dot #* distances[mask]

    shepValue = b[:,0] / (shep + 1e-8)
    shepProjection = shepValue + torch.einsum('ij,ij->i', relPos, b_grad + 1e-8)
    # defaultValue = torch.zeros_like(shepValue)

    projected_q[neighCounts < neighborThreshold] = shepProjection[neighCounts < neighborThreshold]
    projected_q[neighCounts == 0] = defaultValue

#     projected_q = shepProjection

    q_new = q.clone()
    q_new[mask] = projected_q
    return q_new#, queryPositions, shepValue, neighCounts

def liuMirror(
        q: torch.Tensor,
        sim_config: Any,
        scheme_config: Any,
        particles: Any,
        distances: torch.Tensor,
        normals: torch.Tensor,
        current_time: torch.Tensor,
        dt: torch.Tensor,
        neighborThreshold: int = 4,
        defaultValue: float = 0.0,
):
    if len(q.shape) > 1:
        q_flat = q.view(q.shape[0], -1)
        defaults = torch.full((q_flat.shape[1],), defaultValue, device=q.device, dtype=q.dtype) if isinstance(defaultValue, (int, float)) else defaultValue.flatten()
        extended_q = torch.zeros_like(q_flat)
        for i in range(q_flat.shape[1]):
            extended_q[:, i] = liuMirror(
                q = q_flat[:, i],
                sim_config = sim_config,
                scheme_config = scheme_config,
                particles = particles,
                distances = distances,
                normals = normals,
                current_time = current_time,
                dt = dt,
                neighborThreshold = neighborThreshold,
                defaultValue = defaults[i] if defaults.ndim > 0 else defaults.item(),
            )
        return extended_q.view(q.shape)
    mask = torch.logical_and(torch.abs(distances) < particles.supports * 2.0, distances < 0)
    masked_positions = particles.positions[mask]
    relPos = 2 * distances[mask].unsqueeze(1) * normals[mask]
    queryPositions = masked_positions - relPos

    shep, b, A_g, neighCounts = computeLiuMatricesWarp(
        queryPositions = queryPositions,
        referenceParticles = particles,
        referenceQuantities = q,
        operationProperties = OperationProperties(
                kernel = sim_config.kernel,
                supportMode = SupportScheme.Scatter,
                operationMode = OperationDirection.FluidToBoundary,
        ),
        domain = sim_config.domain,
    )
    # print(shep)

    A_g_inv = torch.zeros_like(A_g)
    A_g_inv[neighCounts > 4] = torch.linalg.pinv(A_g[neighCounts > 4])

    res = torch.matmul(A_g_inv, b.unsqueeze(2))[:,:,0]

    extended_q = torch.zeros_like(masked_positions[:,0])

    b_scalar = b[:,0]
    b_grad = b[:,1:]

    res_scalar = res[:,0]
    res_grad = res[:,1:]

    # print(res_scalar, res_grad)

    # directions = torch.nn.functional.normalize(normals[mask], dim=1)
    dot = torch.einsum('ij,ij->i', relPos, res_grad)
    projected_q = res_scalar# + dot #* distances[mask]

    shepValue = b[:,0] / (shep + 1e-8)
    shepProjection = shepValue# + torch.einsum('ij,ij->i', relPos, b_grad + 1e-8)
    # defaultValue = torch.zeros_like(shepValue)

    projected_q[neighCounts < neighborThreshold] = shepProjection[neighCounts < neighborThreshold]
    projected_q[neighCounts == 0] = defaultValue

#     projected_q = shepProjection

    q_new = q.clone()
    q_new[mask] = projected_q
    return q_new#, queryPositions, shepValue, neighCounts