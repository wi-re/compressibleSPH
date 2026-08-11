import random
from ..geometry import ParticleSet
from ..utils.support import volumeToSupport
import torch

def sampleShell(nx, domain, targetNeighbors, circle = True, extraRings = 0):
    dx = (domain.max[0] - domain.min[0]) / nx
    # dx = dx.to(torch.float64)
    area = dx ** 2

    # print(area)
    # 
    # pi dr² = dx^2
    # dr = sqrt(dx^2/pi)
    dr_init = torch.sqrt(dx ** 2 / np.pi)
    dr_init = dx/2

    shells = []

    # shells.append((dr_init, torch.tensor([[0,0]], device = dx.device, dtype = dx.dtype), torch.tensor([dx**2], device = dx.device, dtype = dx.dtype)))

    # print(shells)
    remaining = extraRings
    for i in range(1, nx * 2):
        dr_hat = dr_init
        r_beg = torch.max(torch.hstack([s[1] for s in shells])) if len(shells) > 0 else dr_init
        r_end = r_beg + dr_hat

        shellArea = (np.pi * r_end ** 2 - np.pi * r_beg ** 2)
        # print(f'{i} : n_hat = {shellArea / area}')
        n_hat = torch.ceil(shellArea / area).to(torch.int32)

        optimalArea = n_hat * dx**2

        # pi (r_beg + dr)^2 - pi r_beg^2 = n_hat dx^2

        dr = torch.sqrt(r_beg**2 + n_hat * dx**2 / np.pi) - r_beg

        shellAreaOptimized = (np.pi * (r_beg + dr) ** 2 - np.pi * r_beg ** 2)
        # print(f'{i} : {shellAreaOptimized / optimalArea}, optimal Area: {optimalArea}, shell Area: {shellAreaOptimized}, {dr/ dr_hat}')

        dTheta = 2 * np.pi / n_hat
        theta = torch.linspace(0, 2 * np.pi, n_hat + 1, dtype = dx.dtype, device = dx.device)[:-1]

        theta += random.random() * dTheta*0

        x = (r_beg + dr_hat / 2) * torch.cos(theta)
        y = (r_beg + dr_hat / 2) * torch.sin(theta)
        pts = torch.stack([x,y], dim = 1)
        # print(pts.shape)
        areas = torch.ones(n_hat, device = dx.device, dtype = dx.dtype) * shellAreaOptimized / n_hat
        if circle:
            if r_beg + dr / 2> (domain.max[0] - domain.min[0]) / 2:
                remaining -= 1
                if (remaining < 0):
                    break
        else:
            insidePtcls = (pts[:, 0] > domain.min[0]) & (pts[:, 0] < domain.max[0]) & (pts[:, 1] > domain.min[1]) & (pts[:, 1] < domain.max[1])
            if not insidePtcls.any():
                remaining -= 1
                if (remaining < 0):
                    break
                # break
            pts = pts[insidePtcls]
            areas = areas[insidePtcls]

        shells.append((r_beg, r_end, dr, pts, areas))
        print(f'{i} : r: {r_beg + dr / 2}, n_hat: {n_hat}, shellAreaOptimized: {shellAreaOptimized}, optimalArea: {optimalArea}, ratio: {shellAreaOptimized / optimalArea}')


    positions = torch.vstack([s[3] for s in shells])
    areas = torch.hstack([s[4] for s in shells])

    # print(shellArea / optimalArea)

    
    supports = volumeToSupport(areas, targetNeighbors, domain.dim)
    # supports = torch.ones_like(positions[:, 0]) * support

    return ParticleSet(
        positions = positions,
        supports = supports,
        masses = areas,
        densities = torch.ones_like(positions[:, 0])
    )


import numpy as np
def sampleShellv2(nr, domain, targetNeighbors):
    L = domain.max[0] - domain.min[0]
    # For the shell sampling we first compute the dr, i.e., the thickness of each shell
    # We want to exclude the center point so we start sampling from dr/2 and then keep adding dr until we reach L/2 - dr/2
    # so we have
    # r_min = dr / 2
    # r_max = L / 2 - dr / 2
    # with nr = (r_max - r_min) / dr + 1
    # so 
    # nr = (L/2 - dr/2 - dr/2) / (dr + 1) = (L/2 - dr) / (dr + 1)
    dr = (L/2) / nr
    r_min = dr / 2
    r_max = L / 2 - dr / 2

    # print(f'dr: {dr}, r_min: {r_min}, r_max: {r_max}')

    # for the particle area consider the total sampled domain from -r_max to r_max, which contains nr*2 particles.
    # so over 2 * r_max we have nr*2 particles, so the distance is 2 * r_max / (nr*2) = r_max / nr, so the area is (r_max / nr)^2
    area = (r_max / nr) ** 2

    # print(f'area: {area}')

    shells = []
    for i in range(nr):
        r_begin = 0 + i * dr
        r_end = r_begin + dr
        shellArea = np.pi * (r_end ** 2 - r_begin ** 2)
        n_hat = torch.floor(shellArea / area).to(torch.int32)
        dTheta = 2 * np.pi / n_hat

        # print(f'{i} : r: {r_begin + dr / 2}, n_hat: {n_hat}, shellArea: {shellArea}, area: {area}, ratio: {shellArea / (n_hat * area)}')

        theta = torch.linspace(0, 2 * np.pi, n_hat + 1, dtype = domain.min.dtype, device = domain.min.device)[:-1]
        x = (r_begin + dr / 2) * torch.cos(theta)
        y = (r_begin + dr / 2) * torch.sin(theta)
        pts = torch.stack([x,y], dim = 1)
        areas = torch.ones(n_hat, device = domain.min.device, dtype = domain.min.dtype) * shellArea / n_hat
        shells.append((i, r_begin, r_end, dr, pts, areas))

    positions = torch.vstack([s[4] for s in shells])
    areas = torch.hstack([s[5] for s in shells])
    supports = volumeToSupport(areas, targetNeighbors, domain.dim)

    return ParticleSet(
        positions = positions,
        supports = supports,
        masses = areas,
        densities = torch.ones_like(positions[:, 0])
    ), shells
