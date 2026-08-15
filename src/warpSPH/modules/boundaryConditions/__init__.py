"""Dirichlet-style boundary conditions applied per-variable from a
`config.boundaryConditions` list of SDF-scoped functions: value overrides,
accumulated forcing, and update-tensor overrides.
"""

from .bcs import enforceDirichlet, computeForcing, enforceUpdates

__all__ = ['enforceDirichlet', 'computeForcing', 'enforceUpdates']