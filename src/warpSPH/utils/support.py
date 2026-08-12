"""Support-radius helpers, re-exported from warpSPHCore.

``warpSPHCore.util.support.volumeToSupport`` used to be scalar-only, so this
module carried a tensor-aware wrapper for the per-particle volume call sites
here (shell sampling, Monaghan adaptive support). Core now dispatches on
``isinstance(volume, torch.Tensor)`` internally, so this is a plain re-export.
"""

from warpSPHCore.util.support import n_h_to_nH, nH_to_n_h, volumeToSupport

__all__ = ['n_h_to_nH', 'nH_to_n_h', 'volumeToSupport']
