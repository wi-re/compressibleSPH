"""Signed-distance function for the Rayleigh-Taylor case's top/bottom
Dirichlet buffer bands: a piecewise-safe ``min(y, L - y)``, positive inside
the ``[0, L]`` domain and equal to the distance to the nearer of the two
walls ``y=0``/``y=L``, negative in the bands themselves (``y<0`` or ``y>L``).
`sample.py` uses it (and its gradient) as the `BoundaryCondition.sdf` for
`rayleighTaylorBC`."""

import torch

__all__ = ['buffer_sdf', 'buffer_sdf_gradient']

def buffer_sdf(positions, L):
    dist = torch.zeros_like(positions[:,0])

    maskA = positions[:,1] < 0
    maskB = positions[:,1] > L
    maskC = torch.logical_and(positions[:,1] >= 0, positions[:,1] <= L/2)
    maskD = torch.logical_and(positions[:,1] > L/2, positions[:,1] <= L)

    dist[maskA] = positions[maskA,1]
    dist[maskB] = L - positions[maskB,1]
    dist[maskC] = positions[maskC,1]
    dist[maskD] = L - positions[maskD,1]
    return dist

def buffer_sdf_gradient(positions, L):
    dist = torch.zeros_like(positions)

    maskA = positions[:,1] < 0
    maskB = positions[:,1] > L
    maskC = torch.logical_and(positions[:,1] >= 0, positions[:,1] <= L/2)
    maskD = torch.logical_and(positions[:,1] > L/2, positions[:,1] <= L)

    dist[maskA,1] = 1
    dist[maskB,1] = -1
    dist[maskC,1] = -1
    dist[maskD,1] = 1
    return dist