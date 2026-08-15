"""Constant downward body force for the Rayleigh-Taylor case, applied through
`sample.py`'s domain-wide `BoundaryCondition` (an SDF of ``-1`` everywhere, so
the forcing function fires on every particle rather than a boundary band)."""

import torch

__all__ = ['gravityForcing']

def gravityForcing(state, cfg, schemeCfg, positions, d, n, t, dt, g):
    masses = state.masses
    dvdt = torch.zeros_like(positions)
    dvdt[:, 1] = -g 
    return dvdt * masses.view(-1,1)

