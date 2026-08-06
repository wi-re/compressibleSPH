from .regular import sampleRegularParticles
import torch
from sphWarpCore import *
from sphWarpCore import *
from .wp_deltaShift import computeDeltaShiftWarp


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

