"""Support-radius helpers.

``warpSPHCore.util.support.volumeToSupport`` is scalar-only -- it reaches for
``math.sqrt``, which raises on a multi-element tensor. Several call sites here
pass a *per-particle* volume tensor (shell sampling, Monaghan adaptive
support), so this module wraps it with a tensor-aware version and re-exports
the rest of the core helpers unchanged.

The formulas are identical to core's; only the dispatch differs. If core ever
grows tensor support, this wrapper can collapse back to a plain re-export.
"""

from typing import Union

import numpy as np
import torch

from warpSPHCore.util.support import n_h_to_nH, nH_to_n_h
from warpSPHCore.util.support import volumeToSupport as _volumeToSupportScalar

__all__ = ['n_h_to_nH', 'nH_to_n_h', 'volumeToSupport']


def volumeToSupport(volume: Union[float, torch.Tensor],
                    targetNeighbors: Union[int, float],
                    dim: int):
    """Support radius for a per-particle volume.

    Accepts a float or a tensor; tensors are handled elementwise and keep their
    dtype/device, so this stays usable inside the sampling and adaptive-support
    paths.
    """
    if not isinstance(volume, torch.Tensor):
        return _volumeToSupportScalar(volume, targetNeighbors, dim)

    if dim == 1:
        # N_h = 2 h / v -> h = N_h * v / 2
        return targetNeighbors * volume / 2
    if dim == 2:
        # N_h = pi h^2 / v -> h = sqrt(N_h * v / pi)
        return torch.sqrt(targetNeighbors * volume / np.pi)
    # N_h = 4/3 pi h^3 / v -> h = (N_h * v / pi * 3/4)^(1/3)
    return torch.pow(targetNeighbors * volume / np.pi * 3 / 4, 1 / 3)
