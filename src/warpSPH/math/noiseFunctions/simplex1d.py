"""1D OpenSimplex noise. `_noise1` has no dedicated 1-simplex lattice
implementation of its own — it delegates to `simplex2d._noise2` with
`y` pinned to 0 (hence the commented-out `_extrapolate1`/`STRETCH_CONSTANT1`
imports below are unused).
"""

# from .util import _extrapolate1
# from .constants import STRETCH_CONSTANT1, SQUISH_CONSTANT1, NORM_CONSTANT1
from math import floor
from ctypes import c_int64
from numba import njit, prange

from .simplex2d import _noise2

__all__ = ['_noise1']

@njit(cache=True)
def _noise1(x, perm):
    return _noise2(x,0, perm)