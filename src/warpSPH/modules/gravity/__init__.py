"""Body-force gravity terms (directional, point-source, and radial
potential-field), selected and dispatched by `wrapper.computeGravity` based
on `schemeConfig.gravityConfig`.
"""

from .wrapper import computeGravity

__all__ = ['computeGravity']