from .domain import *
import torch
from .support import volumeToSupportHelper

from typing import NamedTuple
class ParticleSet(NamedTuple):
    positions: torch.Tensor
    supports: torch.Tensor

    masses: torch.Tensor
    densities: torch.Tensor
    
@torch.jit.script
@dataclass#(slots=True)
class PointCloud:
    """
    A named tuple containing the positions of the particles and the number of particles.
    """
    positions: torch.Tensor
    supports: torch.Tensor

    def __ne__(self, other: 'PointCloud') -> bool:
        return not self.__eq__(other)
    
def buildPointCloud(nx, domain: DomainDescription = None, targetNeighbors = 16, jitter = 0.0, band = 0, shortEdge = True):
    periodicity = domain.periodic
    dxs = []
    for d in range(domain.dim):
        l = domain.max[d] - domain.min[d]
        dx = l/(nx if periodicity[d] else nx-1)
        # x = torch.linspace(domain.min[d] + offset, domain.max[d] - offset, nx + band * 2, device = domain.min.device, dtype = domain.min.dtype)
        # spaces.append(x)
        dxs.append(dx)

    spaces = []
    if shortEdge:
        dx = torch.min(torch.tensor(dxs))
    else:
        dx = torch.max(torch.tensor(dxs))
    # print(dxs, dx, nx)
    ns = []
    for d in range(domain.dim):
        l = domain.max[d] - domain.min[d]
        nd = (torch.ceil(l/dx)).to(torch.int32)
        dn = l / (nd if periodicity[d] else nd-1)
        offset = dx/2 if periodicity[d] else 0
        offset -= dx * band
        x = torch.linspace(domain.min[d] + offset, domain.max[d] - offset, nd + band * 2, device = domain.min.device, dtype = domain.min.dtype)
        spaces.append(x)
        ns.append(nd + band * 2)

    # print(f'{shortEdge}: dxs: {dxs}, ns: {ns}, nx: {nx}')




        # print(f'dim: {d}, nx: {nx}, dx: {dx}, min: {x.min()}, max: {x.max()}, periodic: {periodicity[d]}, dxActual: {x[1] - x[0]}')
    # print(dxs)
    grid = torch.meshgrid(*spaces, indexing='ij')
    pos = torch.stack([g.flatten() for g in grid], dim=1)
    # mean_dx = torch.mean(torch.tensor(dxs))
    if jitter > 0:
        pos += torch.rand_like(pos) * dx * jitter
    area = dx ** domain.dim
    support = volumeToSupportHelper(area, targetNeighbors, domain.dim)
    supports = torch.ones_like(pos[:, 0]) * support
    return PointCloud(positions = pos, supports = supports), area, support


def sampleRegularParticles(nx : int, domain : DomainDescription, targetNeighbors: int, jitter = 0.0, band = 0, shortEdge=True):
    pc, area, support = buildPointCloud(nx, domain, targetNeighbors, jitter = jitter, band = band, shortEdge = shortEdge)
    return ParticleSet(positions = pc.positions, supports = pc.supports, masses = torch.ones_like(pc.positions[:, 0]) * area, 
    densities = torch.ones_like(pc.positions[:, 0]))



# from waves.utils.sampling import sampleRegularParticles
# from waves.utils.support import n_h_to_nH

from sphWarpCore import *

from sphWarpCore.radiusSearch.verlet import *
from sphWarpCore.radius import AdjacencyList
from sphWarpCore.operations import *
from sphWarpCore.enumTypes import *

from sphWarpCore import *

from .wp_deltaShift import computeDeltaShiftWarp
# from waves.utils.sampling import ParticleSet


def sampleOptimal(nx, domain, targetNeighbors, kernel, jitter = 0.1, shiftIters = 128, shiftScheme = 'Delta'):


    dim = domain.dim
    device = domain.min.device
    # targetNeighbors = n_h_to_nH(4, dim)

    particles = sampleRegularParticles(
                nx = nx,
                targetNeighbors=targetNeighbors,
                domain=domain,
            ) 

    particleDx = particles.masses.pow(1/dim).mean().item()
    particles = particles._replace(
        positions = particles.positions + jitter * torch.randn_like(particles.positions) * particleDx
    )

    particles = particles._replace(
        positions = torch.rand(particles.positions.shape, device=device) * (domain.max - domain.min) + domain.min
    )

    # print(particles)

    # particles = particles._replace(densities = warpOperation(
    #     particles,
    #     operationProperties = OperationProperties(
    #         operation=WarpOperation.Density,
    #         kernel = KernelFunctions.Wendland2, 
    #         supportMode = SupportScheme.Gather
    #     ),
    #     domain = domain,
    #     adjacency = None
    # ))




    # for i in tqdm(range(shiftIters), leave = False):
    for i in range(shiftIters):
            
        adjacency = radiusSearchCompactHashMap(
            particles, domain, 
            mode = SupportScheme.SuperSymmetric,
            hashMapLengthMode = HashMapLengthMode.Fixed, fixedHashMapLength = 4096
        )


        particles = particles._replace(densities = warpOperation(
            particles,
            operationProperties = OperationProperties(
                operation=WarpOperation.Density,
                kernel = KernelFunctions.Wendland2, 
                supportMode = SupportScheme.Gather
            ),
            domain = domain,
            adjacency = adjacency
        ))


        shift = computeDeltaShiftWarp(
            particles,
            operationProperties = OperationProperties(
                operation=WarpOperation.Density,
                kernel = KernelFunctions.Wendland2, 
                supportMode = SupportScheme.Gather
            ),
            domain = domain,
            # supportMode = SupportScheme.Gather,
            # kernel = KernelFunctions.Wendland2,
            # operationMode = OperationDirection.AllToAll,
            adjacency = adjacency,

            CFL = 0.3, computeMach = False, c_max = 0.3,
            rho0 = 1.0, dx = particleDx,
        )

        # print(f'Iteration {i}, shift magnitude: {shift.norm(dim=1).mean().item()}')
        particles = particles._replace(
            positions = particles.positions + shift
        )
    return ParticleSet(
        positions = particles.positions,
        supports = particles.supports,
        masses = particles.masses,
        densities = particles.densities
    )


from .support import n_h_to_nH
# from ..config import SimulationConfig

class SamplingScheme(Enum):
    regular = 1
    jittered = 2
    glass = 3
    optimal = 4
    random = 5
