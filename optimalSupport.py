from wp_omega import computeOmegaWarp
from compSystem import *
from sphWarpCore import *
from compressibleSPH.config import SimulationConfig
from compressibleSPH.utils.support import volumeToSupportHelper, nH_to_n_h

def computeH(rho, m, targetNeighbors, dim):
    V = m / rho
    return volumeToSupportHelper(V, targetNeighbors, dim)
    return targetNeighbors / 2 * V

def F(h, rho, m, targetNeighbors, dim):
    return h - computeH(rho, m, targetNeighbors, dim)

def evaluateOptimalSupport(
        particleState: CompressibleState,
        config: SimulationConfig,
        supportScheme: SupportScheme = SupportScheme.Scatter,
):
    rhos = [particleState.densities]
    supports = [particleState.supports]

    verletScale = 2**(1/particleState.positions.shape[1])
    nIter = 16
    hThreshold = 1e-3

    hMin = particleState.supports.min()
    hMax = particleState.supports.max()

    iterState = particleState.initializeNewState()
    adjacency = None

    for i in range(nIter):
        adjacency = buildVerletList(iterState, domain = config.domain, verletScale = verletScale, supportMode = supportScheme, priorNeighborhood=adjacency)

        iterState.densities = warpOperation(
            iterState,
            OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Density,
                supportMode = supportScheme,
            ),
            domain = config.domain,
            adjacency=adjacency
        )
        h_prev = iterState.supports

        F_ = F(h_prev, iterState.densities, iterState.masses, targetNeighbors = config.targetNeighbors, dim = config.dim)
        dFdh_ = computeOmegaWarp(iterState, 
                OperationProperties(
                    kernel = config.kernel,
                    supportMode = supportScheme,
                ),
                domain = config.domain,
                adjacency=adjacency)

        h_new = h_prev - F_ / (dFdh_ + 1e-6)

        h_new = h_new.clamp(min = hMin * 0.25, max = hMax * 4.0)
        hMin = h_new.min()
        hMax = h_new.max()

        h_diff = h_new - h_prev
        h_ratio = h_new / (h_prev + 1e-6)
        iterState.supports = h_new

        rhos.append(iterState.densities)
        supports.append(iterState.supports)
        # print(f'Iteration: {i} | Support: {h_new.min()} | {h_new.max()} | {h_new.mean()} | Ratio: {h_ratio.min()} | {h_ratio.max()} | {h_ratio.mean()}')
        if (h_ratio - 1).abs().max() < hThreshold:
            # print('Stopping Early')
            break

    return rhos[-1], supports[-1], rhos, supports