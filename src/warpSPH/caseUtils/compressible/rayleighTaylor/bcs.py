"""Analytic Rayleigh-Taylor profiles used both to seed the initial state and
as the Dirichlet target functions of `sample.py`'s buffer-band boundary
condition, so the top/bottom bands stay pinned to the unperturbed
stratification for the run's duration.

`rayleighTaylor_rho` is a smooth (logistic, width `delta`) interpolation
between the bottom density `rho_b` and top density `rho_t` in `y`, centred at
``y = 0.5``. `rayleighTaylor_u` derives the matching internal energy from a
hydrostatic pressure profile (``dP/dy = -g*rho``, integrated with constant
`rho_y` rather than the true stratified density -- an approximation that is
fine for the shallow buffer bands this feeds, not the bulk flow) via the
ideal-gas relation ``u = P / (rho * (gamma - 1))``. The `RayleighTaylor*`
functions adapt these to the boundary-condition callback signature
(`positions` in, `torch.Tensor` out); `RayleighTaylorVelocity` and
`RayleighTaylorAcceleration` both pin to zero, holding the bands fixed.
"""

import torch

__all__ = ['rayleighTaylor_rho', 'RayleighTaylorVelocity', 'RayleighTaylorAcceleration',
           'RayleighTaylorDensity', 'RayleighTaylorInternalEnergy']

def rayleighTaylor_rho(positions, rho_b, rho_t, delta):
    # print('Enforcing Rayleigh-Taylor density for ', positions.shape[0], ' particles')
    y = positions[:,1]
    rho_y = rho_b + (rho_t - rho_b) * (1 + torch.exp(-(y - 0.5) / delta))**(-1)
    return rho_y
def rayleighTaylor_u(positions, rho_b, rho_t, delta, g, gamma):
    y = positions[:,1]
    rho_y = rayleighTaylor_rho(positions, rho_b, rho_t, delta)
    P_0 = rho_t / gamma
    P = P_0 - g * rho_y * (y - 1/2)
    u = P / rho_y / (gamma - 1)
    return u

# BC functions

def RayleighTaylorVelocity(positions):
    return torch.zeros_like(positions)

def RayleighTaylorAcceleration(positions):
    return torch.zeros_like(positions)

def RayleighTaylorDensity(positions, rho_b, rho_t, delta):
    return rayleighTaylor_rho(positions, rho_b, rho_t, delta)

def RayleighTaylorInternalEnergy(positions, rho_b, rho_t, delta, g, gamma):
    u = rayleighTaylor_u(positions, rho_b, rho_t, delta, g, gamma)
    return u
