import torch

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
