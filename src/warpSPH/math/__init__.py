"""Math helpers subpackage: periodic-boundary position wrapping
(`getPeriodicPositions`), the Perlin/simplex noise generator (`noise.py`,
`noiseFunctions/`), and a portable scatter-reduction helper (`scatter.py`)
vendored from PyTorch Geometric so warpSPH doesn't depend on it.
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
from .scatter import scatter_sum, broadcast

__all__ = ['getPeriodicPositions', 'generateNoise', 'generateOctaveNoise', 'scatter_sum', 'broadcast']