import torch
from dataclasses import dataclass
import numpy as np

@torch.jit.script
@dataclass(slots=True)
class DomainDescription:
    """
    A named tuple containing the minimum and maximum domain values.
    """
    min: torch.Tensor
    max: torch.Tensor
    periodic: torch.Tensor
    dim: int

    def __ne__(self, other: 'DomainDescription') -> bool:
        return not self.__eq__(other)
    
def buildDomainDescription(l, dim, periodic = False, device = 'cpu', dtype = torch.float32):
    minDomain = [-l/2] * dim
    maxDomain = [l/2] * dim
    return DomainDescription(torch.tensor(minDomain, device = device, dtype = dtype), torch.tensor(maxDomain, device = device, dtype = dtype), torch.tensor([periodic] * dim, dtype = torch.bool, device = device) if isinstance(periodic,bool) else periodic, dim)
