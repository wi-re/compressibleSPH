import torch

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