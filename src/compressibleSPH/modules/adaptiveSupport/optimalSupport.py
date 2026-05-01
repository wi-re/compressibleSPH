from .wp_omega import computeOmegaWarp
from ...systems.baseState import *
from sphWarpCore import *
from compressibleSPH.config import SimulationConfig
from compressibleSPH.utils.support import volumeToSupportHelper, nH_to_n_h
from torch.profiler import profile, record_function, ProfilerActivity

def computeH(rho, m, targetNeighbors, dim):
    V = m / rho
    return volumeToSupportHelper(V, targetNeighbors, dim)
    return targetNeighbors / 2 * V

def F(h, rho, m, targetNeighbors, dim):
    return h - computeH(rho, m, targetNeighbors, dim)
from diffSPH.schemes.states.compressiblesph import CompressibleState as CompState
from diffSPH.neighborhood import evaluateNeighborhood
from diffSPH.enums import KernelType
from diffSPH.kernels import getSPHKernelv2
from diffSPH.modules.density import computeDensity as computeDensityDiffSPH
from diffSPH.modules.adaptiveSmoothing import computeOmega as computeOmegaDiffSPH
from sphWarpCore.enumTypes import SupportScheme
from diffSPH.enums import SupportScheme as DiffSPHSupportScheme

def evaluateOptimalSupport(
        particleState: BaseState,
        config: SimulationConfig,
        supportScheme: SupportScheme = SupportScheme.Scatter,
        adjacency: Optional[AdjacencyList] = None,
):
    with record_function("evaluateOptimalSupport"):
        rhos = [particleState.densities]
        supports = [particleState.supports]

        verletScale = 2**(1/particleState.positions.shape[1])
        nIter = 16
        hThreshold = 1e-3

        hMin = particleState.supports.min()
        hMax = particleState.supports.max()

        iterState = particleState.initializeNewState()
        # adjacency = None

        for i in range(nIter):
            with record_function(f"[evalOS] Iteration {i}"):
                with record_function("[evalOS] buildVerletList"):
                    adjacency = buildVerletList(iterState, domain = config.domain, verletScale = verletScale, supportMode = SupportScheme.SuperSymmetric, priorNeighborhood=adjacency, verbose=False)

                                    
                # diffSPHState = CompState(
                #     positions = iterState.positions,
                #     velocities = iterState.velocities,
                #     densities = iterState.densities,
                #     supports = iterState.supports,
                #     internalEnergies = iterState.internalEnergies,
                #     totalEnergies = iterState.totalEnergies,
                #     entropies = iterState.entropies,
                #     soundspeeds= iterState.soundspeeds,
                #     masses = iterState.masses,
                #     kinds = iterState.kinds,
                #     materials = iterState.materials,
                #     UIDs = iterState.UIDs,
                #     pressures = iterState.pressures,
                #     omega = None,
                # )

                # kernel_ = KernelType.Wendland2
                # wrappedKernel = getSPHKernelv2(KernelType.Wendland2)
                # verletScale = 2 ** (1/config.dim)
                # verletScale = 1

                # neighborhood, neighbors = evaluateNeighborhood(diffSPHState, config.domain, KernelType.Wendland2, verletScale = verletScale, mode = DiffSPHSupportScheme.SuperSymmetric, priorNeighborhood=None)



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
                # iterState.densities = computeDensityDiffSPH(diffSPHState, kernel_, neighbors.get('noghost'), DiffSPHSupportScheme.Gather, config)

                h_prev = iterState.supports

                F_ = F(h_prev, iterState.densities, iterState.masses, targetNeighbors = config.targetNeighbors, dim = config.dim)
                dFdh_ = computeOmegaWarp(iterState, 
                        OperationProperties(
                            kernel = config.kernel,
                            supportMode = supportScheme,
                        ),
                        domain = config.domain,
                        adjacency=adjacency)
                # dFdh_ = computeOmegaDiffSPH(diffSPHState, kernel_, neighbors.get('noghost'), DiffSPHSupportScheme.Gather, config)

                h_new = h_prev - F_ / (dFdh_ + 1e-6)

                h_new = h_new.clamp(min = hMin * 0.25, max = hMax * 4.0)
                hMin = h_new.min()
                hMax = h_new.max()

                h_diff = h_new - h_prev
                h_ratio = h_new / (h_prev + 1e-6)
                iterState.supports = h_new

                rhos.append(iterState.densities)
                supports.append(iterState.supports)
                        
                # print(f'Iteration: {i} | h_ratio: {h_ratio.min()} | {h_ratio.max()} | {h_ratio.mean()}')
                # print(f'Densities: {iterState.densities.min()} | {iterState.densities.max()} | {iterState.densities.mean()}')
                # print(f'Supports: {h_new.min()} | {h_new.max()} | {h_new.mean()}')
                # print(f'Iteration: {i} | Support: {h_new.min()} | {h_new.max()} | {h_new.mean()} | Ratio: {h_ratio.min()} | {h_ratio.max()} | {h_ratio.mean()}')
                if (h_ratio - 1).abs().max() < hThreshold:
                    # print('Stopping Early')
                    break

        return rhos[-1], supports[-1], adjacency, rhos, supports