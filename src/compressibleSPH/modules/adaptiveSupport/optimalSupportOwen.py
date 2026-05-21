from ...systems import CompressibleState
from ...config import SimulationConfig, CompressibleSPHConfig
import numpy as np
import torch
from sphWarpCore import *
from typing import Optional, Union

from .wp_psi0 import computePsi0Warp
from .owenLUT import computeOwen, interpolateLUT

PsiLUT_fn = None

# assumes convention from CRKSPH with $\eta$ = 1
def n_h_to_nH(n_h, dim):
    spacing = 1 / n_h
    v = spacing**dim
    vH = 2.0 if dim == 1 else (np.pi if dim == 2 else (4/3) * np.pi)
    return vH / v

def nH_to_n_h(nH, dim):
    vH = 2.0 if dim == 1 else (np.pi if dim == 2 else (4/3) * np.pi)
    v = vH / nH
    return (1 / v)**(1/dim)


def computeNewSupport(target, n_h, h):        
    n_h_Target = target
    s = n_h_Target / n_h
    a = torch.where(s >= 1.0, 0.4 * (1 + s**-3), 0.4 * (1 + s**2))
    h_i_new = (1 - a + a * s) * h
    # h_i_new = s * h
    return h_i_new

def evaluateOptimalSupportOwen(
        particles: CompressibleState,
        config: SimulationConfig,
        compConfig: CompressibleSPHConfig,
        kernel_: Optional[KernelFunctions] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None,
        supportScheme: Optional[SupportScheme] = None,
        verbose = False
        ):
    global PsiLUT_fn

    kernel = kernel_ if kernel_ is not None else config.kernel
    if PsiLUT_fn is None:
        PsiLUT_fn = computeOwen(kernel_, dim = config.domain.dim, nMin = 2.0, nMax = 6.0, nLUT = 1024)
        # config['support']['LUT'] = PsiLUT_fn
    # particles, domain, kernel, targetNeighbors, PsiLUT_fn, nIter = 16, neighborhood = None, verbose = False,eps = 1e-3, neighborhoodAlgorithm = 'compact'):
    hs = [particles.supports]
    # print(particles)
    psis = []

    supportMode = config.supportMode if supportScheme is not None else SupportScheme.SuperSymmetric
    targetNeighbors = config.targetNeighbors
    dim = config.domain.dim
    
    nhTarget = nH_to_n_h(targetNeighbors, dim = dim)
    verletScale = config.verletScale
    # verletScale = 1.0 if config is None else (config['neighborhood']['verletScale'] if 'verletScale' in config else 1.0)
    hMin = particles.supports.min()
    hMax = particles.supports.max()

    nIter = compConfig.adaptiveSupportIterations
    adaptiveHThreshold = compConfig.adaptiveSupportThreshold
    # verbose = False
    
    for i in range(nIter):
        if verbose:
            print('----------------------------------')
            print(f'Iteration {i}, target: {nhTarget}')
        adjacency = buildVerletList(particles, domain = config.domain, verletScale = verletScale, supportMode = SupportScheme.SuperSymmetric, priorNeighborhood=adjacency, verbose=False)

        
        # psi_0, psi_0_H = computePsi_0(particles, kernel_, neighbors.get('noghost'), supportScheme, config)
        psi_0, psi_0_H = computePsi0Warp(
            particles,
            operationProperties=OperationProperties(
                kernel = config.kernel,
                supportMode = SupportScheme.Gather
            ),
            domain = config.domain,
            adjacency = adjacency
        )
        if verbose:
            print(f'Psi: {psi_0_H.min()} | {psi_0_H.max()} | {psi_0_H.mean()}')
            
        n_h_i = PsiLUT_fn.fromPsiH(psi_0_H)
        if verbose:
            print(f'n_h: {n_h_i.min()} | {n_h_i.max()} | {n_h_i.mean()}')
        
        h = computeNewSupport(nhTarget, n_h_i, particles.supports)
        psis.append(psi_0_H)
        h = h.clamp(min = hMin * 0.25, max = hMax * 4.0)
        hMin = h.min()
        hMax = h.max()
        
        h_ratio = h / particles.supports
        
        if verbose: 
            print(f'Support: {h.min()} | {h.max()} | {h.mean()}')
            print(f'Ratio: {(h_ratio).min()} | {(h_ratio).max()} | {(h_ratio).mean()}')
            
        particles.supports = h

        hs.append(h)

        if (h_ratio - 1).abs().max() < adaptiveHThreshold:
            if verbose: 
                print('Stopping Early')
            # print('Stopping Early')
            break

    densities = warpOperation(
        particles,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Density,
            supportMode = supportScheme,
        ),
        domain = config.domain,
        adjacency = adjacency
    )
    
    return densities, hs[-1], adjacency, psis, hs