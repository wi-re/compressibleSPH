"""Particle shifting technique (PST): anti-clustering position correction,
with optional free-surface projection of the shift. The shift schemes
(`ShiftingScheme` in `configurations.moduleConfigurations.shifting`) are:
explicit delta-SPH (`computeDeltaShiftWarp`/`computeDeltaShift`); implicit
particle shifting (`computeImplicitShift`), a matrix-free Krylov
(BiCGStab/GMRES) solve over the neighbor graph; and `dynamic`
(`computeDynamicImplicitShift`), the same implicit path with the opt-in
inner-solve fallback chain (`ShiftProperties.implicitFallback`) enabled by
default -- the improved scheme, while `implicit` stays byte-identical for
legacy users.
"""

from .delta import computeDeltaShiftWarp
from .implicitShifting import computeImplicitShift, computeDynamicImplicitShift
from .wrapper import solveShifting

__all__ = [
    'computeDeltaShiftWarp',
    'computeImplicitShift',
    'computeDynamicImplicitShift',
    'solveShifting'
]