"""Extract a region's boundary as 2D polylines, for plotting. Evaluates `sdf`
on a regular grid spanning `config.domain` and runs marching squares
(`skimage.measure.find_contours`) at the given `level`; 2D only. Grid
resolution is derived from `nGrid` divided along the domain's shorter
(`minOrMax='min'`) or longer (`'max'`) side, so the two axes can get
different point counts.
"""

from typing import Optional
from skimage import measure
import torch

__all__ = ['find_contour']


def find_contour(config, schemeConfig, sdf, nGrid, level = 0, minOrMax = 'min'):
    minExtent = config.domain.min.cpu()
    maxExtent = config.domain.max.cpu()
    Lx = config.domain.max[0] - config.domain.min[0]
    Ly = config.domain.max[1] - config.domain.min[1]
    if minOrMax == 'min':
        minL = min(Lx, Ly)
    else:
        minL = max(Lx, Ly)
        
    dL = minL / nGrid
    nx = (Lx//dL).to(torch.int32).cpu().item()
    ny = (Ly//dL).to(torch.int32).cpu().item()

    # print(nx, ny, ngrid)

    # x = torch.linspace(config['domain'].min[0]-Lx * 0.01, config['domain'].max[0] + Lx*0.01, nx, dtype = torch.float32)
    # y = torch.linspace(config['domain'].min[1]-Ly * 0.01, config['domain'].max[1] + Ly*0.01, ny, dtype = torch.float32)
    x = torch.linspace(config.domain.min[0], config.domain.max[0], nx, dtype = torch.float32)
    y = torch.linspace(config.domain.min[1], config.domain.max[1], ny, dtype = torch.float32)
    X, Y = torch.meshgrid(x, y, indexing = 'ij')
    P = torch.stack([X,Y], dim=-1)
    domain = config.domain
    points = P.reshape(-1,2).to(domain.min.device).to(domain.min.dtype)
    f = sdf(points)[0].reshape(nx, ny).cpu()
    contours = measure.find_contours(f.numpy(), level)
    for ic in range(len(contours)):
        contours[ic][:,0] = (contours[ic][:,0]) / (f.shape[0] - 1) * (maxExtent[0] - minExtent[0]).numpy() + minExtent[0].numpy()
        contours[ic][:,1] = (contours[ic][:,1]) / (f.shape[1] - 1) * (maxExtent[1] - minExtent[1]).numpy() + minExtent[1].numpy()
    return contours
