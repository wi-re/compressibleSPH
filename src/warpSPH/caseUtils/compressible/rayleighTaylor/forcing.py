import torch

def gravityForcing(state, cfg, schemeCfg, positions, d, n, t, dt, g):
    masses = state.masses
    dvdt = torch.zeros_like(positions)
    dvdt[:, 1] = -g 
    return dvdt * masses.view(-1,1)

