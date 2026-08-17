"""Math helpers subpackage: periodic-boundary position wrapping
(`getPeriodicPositions`), the Perlin/simplex noise generator (`noise.py`,
`noiseFunctions/`), a device-resident linear grid interpolator
(`interpolation.py`, a torch port of scipy's `RegularGridInterpolator`), and a
portable scatter-reduction helper (`scatter.py`) vendored from PyTorch Geometric
so warpSPH doesn't depend on it.
"""

import torch

def getPeriodicPositions(x, domain):
    minD = domain.min.detach().to(x.device)
    maxD = domain.max.detach().to(x.device)
    periodicity = domain.periodic
    pos = [(torch.remainder(x[:, i] - minD[i], maxD[i] - minD[i]) + minD[i]) if periodicity[i] else x[:,i] for i in range(domain.dim)]
    modPos = torch.stack(pos, dim = -1)
    return modPos

from .noise import generateNoise, generateOctaveNoise
from .interpolation import RegularGridInterpolator
from .scatter import scatter_sum, broadcast

__all__ = ['getPeriodicPositions', 'generateNoise', 'generateOctaveNoise', 'RegularGridInterpolator', 'scatter_sum', 'broadcast']