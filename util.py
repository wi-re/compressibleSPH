import torch
from sphWarpCore import *

def mergeParticles(particles_l, particles_r):
    positions = torch.cat([particles_l.positions, particles_r.positions], dim = 0)
    supports = torch.cat([particles_l.supports, particles_r.supports], dim = 0)
    masses = torch.cat([particles_l.masses, particles_r.masses], dim = 0)
    densities = torch.cat([particles_l.densities, particles_r.densities], dim = 0)

    sortedIndices = torch.argsort(positions[:, 0])
    positions = positions[sortedIndices]
    supports = supports[sortedIndices]
    masses = masses[sortedIndices]
    densities = densities[sortedIndices]

    return ParticleState(positions, supports, masses, densities, None)

def plotToAxis1D(ax, positions, values, title = '', **kwargs):
    order = torch.argsort(positions[:, 0])

    ax.scatter(positions.cpu(), values.cpu(), s = 1, **kwargs)
    ax.set_title(title)
