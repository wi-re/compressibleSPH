"""Signed-distance-function primitives and combinators.

Re-exported through ``warpSPH.sampling.sdf``; kept a regular package so wheel
packaging does not depend on namespace-package discovery.
"""

from .implicitFunctions import *
from .operators import *

__all__ = []
__all__.extend(implicitFunctions.__all__)
__all__.extend(operators.__all__)