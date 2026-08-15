"""Geometry API: signed-distance-function primitives/combinators (`sdf`,
`sdfFunctionality`), NACA 4-/5-digit airfoil boundary generation (`naca`),
and the particle/point-cloud data types (`types`) that regions and samplers
build on.
"""

from .naca import generate_naca_airfoil, eval_distance, eval_naca

from .sdf import *
from .types import ParticleState, ParticleSet, PointCloud, SamplingScheme

__all__ = []
__all__.extend(['generate_naca_airfoil', 'eval_distance', 'eval_naca'])
__all__.extend(sdf.__all__)
__all__.extend(['ParticleState', 'ParticleSet', 'PointCloud', 'SamplingScheme'])