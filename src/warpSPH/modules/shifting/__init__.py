"""Particle shifting technique (PST): anti-clustering position correction,
with optional free-surface projection of the shift. Two shift schemes are
available (`ShiftingScheme` in `configurations.moduleConfigurations.shifting`):
explicit delta-SPH (`computeDeltaShiftWarp`/`computeDeltaShift`) and implicit
particle shifting (`computeImplicitShift`), a matrix-free BiCGStab solve over
the neighbor graph.
"""

from .delta import computeDeltaShiftWarp
from .implicitShifting import computeImplicitShift
from .wrapper import solveShifting

__all__ = [
    'computeDeltaShiftWarp',
    'computeImplicitShift',
    'solveShifting'
]