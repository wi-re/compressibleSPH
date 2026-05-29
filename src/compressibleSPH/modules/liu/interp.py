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
):

    b, A_g, neighCounts = computeLiuMatricesWarp(
        queryPositions = queryPositions,
        referenceParticles = referenceParticles,
        referenceQuantities = referenceQuantities,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = SupportScheme.Scatter,
        ),
        domain = config.domain,
    )

    A_g_inv = torch.zeros_like(A_g)
    A_g_inv[neighCounts > neighbor_threshold] = torch.linalg.pinv(A_g[neighCounts > neighbor_threshold])

    res = torch.matmul(A_g_inv, b.unsqueeze(2))[:,:,0]

    return res[:,0], res[:,1:], neighCounts
