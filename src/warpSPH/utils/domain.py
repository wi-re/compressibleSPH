"""`buildDomainDescription`, the one constructor used across cases/configs to
turn a domain side length into a centered-at-origin `warpSPHCore.
DomainDescription` (min/max/periodic/dim tensors). `DomainDescription` itself
now lives in `warpSPHCore` -- the commented-out dataclass below is the
pre-move local definition, kept as a record rather than deleted.
"""

import torch
from dataclasses import dataclass
import numpy as np
from warpSPHCore import DomainDescription

__all__ = ['DomainDescription', 'buildDomainDescription']

# @torch.jit.script
# @dataclass(slots=True)
# class DomainDescription:
#     """
#     A named tuple containing the minimum and maximum domain values.
#     """
#     min: torch.Tensor
#     max: torch.Tensor
#     periodic: torch.Tensor
#     dim: int

#     def __ne__(self, other: 'DomainDescription') -> bool:
#         return not self.__eq__(other)
    
def buildDomainDescription(l, dim, periodic = False, device = 'cpu', dtype = torch.float32):
    minDomain = [-l/2] * dim
    maxDomain = [l/2] * dim
    return DomainDescription(torch.tensor(minDomain, device = device, dtype = dtype), torch.tensor(maxDomain, device = device, dtype = dtype), torch.tensor([periodic] * dim, dtype = torch.bool, device = device) if isinstance(periodic,bool) else periodic, dim)
