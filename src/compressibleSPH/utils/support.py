import torch
from dataclasses import dataclass
import numpy as np
from typing import Union

def n_h_to_nH(n_h, dim):
    spacing = 1 / n_h
    v = spacing**dim
    vH = 2.0 if dim == 1 else (np.pi if dim == 2 else (4/3) * np.pi)
    return vH / v

def nH_to_n_h(nH, dim):
    vH = 2.0 if dim == 1 else (np.pi if dim == 2 else (4/3) * np.pi)
    v = vH / nH
    return (1 / v)**(1/dim)


def volumeToSupportHelper(volume : float, targetNeighbors : Union[int, float], dim : int):
    """
    Calculates the support radius based on the given volume, target number of neighbors, and dimension.

    Parameters:
    volume (float): The volume of the support region.
    targetNeighbors (int): The desired number of neighbors.
    dim (int): The dimension of the space.

    Returns:
    torch.Tensor: The support radius.
    """
    if dim == 1:
        # N_h = 2 h / v -> h = N_h * v / 2
        return targetNeighbors * volume / 2
    elif dim == 2:
        # N_h = \pi h^2 / v -> h = \sqrt{N_h * v / \pi}
        if isinstance(volume, torch.Tensor):
            return torch.sqrt(targetNeighbors * volume / np.pi)
        else:
            return np.sqrt(targetNeighbors * volume / np.pi)
    else:
        # N_h = 4/3 \pi h^3 / v -> h = \sqrt[3]{N_h * v / \pi * 3/4}
        if isinstance(volume, torch.Tensor):
            return torch.pow(targetNeighbors * volume / np.pi * 3 /4, 1/3)
        else:
            return np.pow(targetNeighbors * volume / np.pi * 3 /4, 1/3)