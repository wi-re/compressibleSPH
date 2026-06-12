from ..utils.domain import DomainDescription
from ..utils.sampling import PointCloud, ParticleSet, volumeToSupportHelper

import torch

def buildPointCloud(nx, domain: DomainDescription = None, targetNeighbors = 16, jitter = 0.0, band = 0, shortEdge = True):
    periodicity = domain.periodic
    dxs = []
    for d in range(domain.dim):
        l = domain.max[d] - domain.min[d]
        dx = l/(nx if periodicity[d] else nx-1)
        # x = torch.linspace(domain.min[d] + offset, domain.max[d] - offset, nx + band * 2, device = domain.min.device, dtype = domain.min.dtype)
        # spaces.append(x)
        dxs.append(dx)

    spaces = []
    if shortEdge:
        dx = torch.min(torch.tensor(dxs))
    else:
        dx = torch.max(torch.tensor(dxs))
    # print(dxs, dx, nx)
    ns = []
    for d in range(domain.dim):
        l = domain.max[d] - domain.min[d]
        nd = (torch.ceil(l/dx)).to(torch.int32)
        dn = l / (nd if periodicity[d] else nd-1)
        offset = dx/2 if periodicity[d] else 0
        offset -= dx * band

        # print(f'Dimension {d}: l: {l}, dx: {dx}, nd: {nd}, dn: {dn}, offset: {offset}')
        x = torch.linspace(domain.min[d] + offset, domain.max[d] - offset, nd + band * 2, device = domain.min.device, dtype = domain.min.dtype)
        spaces.append(x)
        ns.append(nd + band * 2)

    # print(f'{shortEdge}: dxs: {dxs}, ns: {ns}, nx: {nx}')



    dtype = spaces[0].dtype
        # print(f'dim: {d}, nx: {nx}, dx: {dx}, min: {x.min()}, max: {x.max()}, periodic: {periodicity[d]}, dxActual: {x[1] - x[0]}')
    # print(dxs)
    grid = torch.meshgrid(*spaces, indexing='xy')
    pos = torch.stack([g.flatten() for g in grid], dim=1)
    # mean_dx = torch.mean(torch.tensor(dxs))
    if jitter > 0:
        pos += torch.rand_like(pos) * dx * jitter
    area = dx ** domain.dim
    support = volumeToSupportHelper(area, targetNeighbors, domain.dim)
    supports = torch.ones_like(pos[:, 0]) * support
    return PointCloud(positions = pos, supports = supports), area, support


def sampleRegularParticles(nx : int, domain : DomainDescription, targetNeighbors: int, jitter = 0.0, band = 0, shortEdge=True):
    pc, area, support = buildPointCloud(nx, domain, targetNeighbors, jitter = jitter, band = band, shortEdge = shortEdge)
    return ParticleSet(positions = pc.positions, supports = pc.supports, masses = torch.ones_like(pc.positions[:, 0]) * area, 
    densities = torch.ones_like(pc.positions[:, 0]))
