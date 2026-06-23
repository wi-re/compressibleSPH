from scipy.interpolate import RegularGridInterpolator
from ..density.density import computeDensities
from sphWarpCore import *
import torch
import numpy as np
from ...utils.noise import generateNoise


def getSpacing(nx, domain: DomainDescription):
    periodicity = domain.periodic
    dxs = []
    for d in range(domain.dim):
        l = domain.max[d] - domain.min[d]
        dx = l/(nx if periodicity[d] else nx-1)
        dxs.append(dx.item())
    return np.min(dxs)

def generateNoiseInterpolator(fluidResolution, noiseResolution, domain: DomainDescription, dim = 2, octaves = 3, lacunarity=2, persistence = 0.5, baseFrequency = 1, tileable = False, kind = 'perlin', seed = 1248097):
    _, _, noiseField = generateNoise(noiseResolution, dim = dim, octaves = octaves, lacunarity=lacunarity, persistence = persistence, baseFrequency = baseFrequency, tileable = tileable, kind = kind, seed = seed)
    grid = []
    dx = 1/2 * getSpacing(fluidResolution, domain)
    for d in range(domain.dim):
        grid.append(np.linspace(domain.min[d].detach().cpu() + dx , domain.max[d].detach().cpu() - dx, noiseResolution))
    interpolator = RegularGridInterpolator(grid, noiseField.cpu().numpy(), bounds_error=False, fill_value=None)
    return lambda x: torch.tensor(interpolator(x.cpu().numpy()).reshape(x.shape[0])).to(x.device)



def sampleDivergenceFreeNoise(particleState, domain, config, schemeConfig, nxGrid, octaves = 3, lacunarity = 2, persistence = 0.5, baseFrequency = 1, tileable = True, kind = 'perlin', seed = 45906734):
    # neighborhood, neighbors = evaluateNeighborhood(particleState, domain, config['kernel'], verletScale = config['neighborhood']['verletScale'], mode =  SupportScheme.SuperSymmetric, priorNeighborhood=None)
    
    adjacency = buildVerletList(
        particleState, 
        config.domain, verletScale = config.verletScale, supportMode = SupportScheme.SuperSymmetric,
        priorNeighborhood = None,
        verbose = False)

    noiseGen = generateNoiseInterpolator(nxGrid, nxGrid, domain, dim = domain.dim, octaves = octaves, lacunarity = lacunarity, persistence = persistence, baseFrequency = baseFrequency, tileable = tileable, kind = kind, seed = seed)

    dtype = particleState.positions.dtype
    # ramp = generateRamp(particleState, config) if len([r for r in config['regions'] if r['type'] == 'boundary']) > 0 else 1
    potential = noiseGen(particleState.positions).to(dtype) #* ramp

    rho = computeDensities(particleState, config, schemeConfig, adjacency)
    priorDensity = particleState.densities.clone() 
    particleState.densities = rho

    gradTerm = warpOperation(
        particleState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Gradient,
            supportMode = SupportScheme.Gather, # cullen switch E.1 in the CRK paper uses gather for density estimation
        ),
        queryValues = potential,
        domain = config.domain,
        adjacency = adjacency,
    )

    # gradTerm = SPHOperation(particleState, potential, config['kernel'], neighbors.get('noghost')[0], neighbors.get('noghost')[1], Operation.Gradient, SupportScheme.Scatter, GradientMode.Difference)
    
    # sph_op(particleState, particleState, domain, config['kernel'], sparseNeighborhood, operation = 'gradient', supportScheme = 'symmetric', quantity = potential, gradientMode = 'difference')

    velocities = torch.stack([gradTerm[:,1], -gradTerm[:,0]], dim = 1)
    velocities = velocities / torch.linalg.norm(velocities, dim = 1, keepdim = True).max()

    velocities[particleState.kinds != 0, :] = 0

    particleState.densities = priorDensity
    return velocities