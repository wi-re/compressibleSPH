"""Internal-energy time derivative for the Monaghan-style compressible
scheme (pressure-velocity-divergence work term).
"""

from .dudt import compute_dudt_warp as computeDudtMonaghan

__all__ = ['computeDudtMonaghan']