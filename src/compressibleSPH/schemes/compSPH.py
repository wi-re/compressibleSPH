from ..modules.adaptiveSupport import computeOmega
from ..modules.compSPH.accel import computeCompSPHAccelWarp
from ..modules.compSPH.dudt import computeCompSPHdudtWarp
from ..modules.compSPH.balance import computeCompSPHBalanceTermWarp
from ..enumTypes import EnergyScheme, ViscositySwitch
from ..modules import *

from sphWarpCore import *
from ..systems.compSPH import CompSPHSystem, CompSPHState
from ..configurations.compSPHConfig import CompSPHConfig
from ..configurations.simulationConfig import SimulationConfig
import torch
from ..systems.compressibleMonaghan import CompressibleSystemUpdate
# from diffSPH.schemes.states.compressiblesph import CompressibleState as CompState
# from diffSPH.kernels import getSPHKernelv2
# from diffSPH.neighborhood import evaluateNeighborhood
# from diffSPH.enums import KernelType as KernelTypeDiffSPH

# from diffSPH.modules.compSPH import compSPH_acceleration, compSPH_dudt, compute_fij
# from diffSPH.enums import EnergyScheme as EnergySchemeDiffSPH

from ..modules.shockCapturing.CullenHopkins import computeHopkinsTerms, computeHopkinsUpdate

lut = None

from diffSPH.simple import *
from diffSPH.modules.adaptiveSmoothingASPH import n_h_to_nH, evaluateOptimalSupportOwen

from diffSPH.enums import EnergyScheme as EnergySchemeDiffSPH
from diffSPH.schemes.states.compressiblesph import CompressibleState as CompState
from diffSPH.kernels import getSPHKernelv2
from diffSPH.neighborhood import evaluateNeighborhood
from diffSPH.modules.density import computeDensity

from diffSPH.modules.compSPH import compSPH_acceleration, compSPH_dudt, compute_fij

from diffSPH.enums import SupportScheme as SupportSchemeDiffSPH
# from sphWarpCore.kernels.wp_kernel import sphKernel
from diffSPH.kernels import Kernel as diffSPHKernel
from diffSPH.kernels import KernelType
from diffSPH.neighborhood import coo_to_csr

from sphWarpCore import *

import time
import torch
# from diffSPH.regions import enforceDirichlet, enforceDirichletUpdate

class TimedBlock:
    def __init__(self, name: str = "Timed block", use_cuda: bool = True, device=None):
        self.name = name
        self.device = device
        self.use_cuda = use_cuda and torch.cuda.is_available()

        self.cpu_ms = None
        self.cuda_ms = None

        self._cpu_start = None
        self._start_event = None
        self._end_event = None

    def __enter__(self):
        self._cpu_start = time.perf_counter()

        if self.use_cuda:
            if self.device is not None:
                torch.cuda.synchronize(self.device)
            else:
                torch.cuda.synchronize()

            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cpu_ms = (time.perf_counter() - self._cpu_start) * 1000.0

        # if self.use_cuda:
        #     self._end_event.record()
        #     self._end_event.synchronize()
        #     self.cuda_ms = self._start_event.elapsed_time(self._end_event)
        #     print(f"[{self.name}] CPU: {self.cpu_ms:.3f} ms | CUDA: {self.cuda_ms:.3f} ms")
        # else:
        #     print(f"[{self.name}] CPU: {self.cpu_ms:.3f} ms | CUDA: N/A")

        return False

def compSPH_step(
    system: CompSPHSystem,
    dt: float,
    config: SimulationConfig,
    compParams: CompSPHConfig,
    verbose = False,
    # dsphConfig = None,
):        
    global lut
    currentSystem = system#
    currentState = currentSystem.state


    # dsphConfig = {    
    #     'diffusion':{
    #         'C_l': 1,
    #         'C_q': 1,
    #         'Cu_l': 1,
    #         'Cu_q': 1,
    #         'monaghanSwitch': True,
    #         'viscosityTerm': 'Monaghan',
    #         'correctXi': True,
            
    #         'viscosityFormulation': 'Monaghan1992',
    #         'thermalConductivityFormulation': 'Price2008',
    #         'signalTerm': 'Price2019',
    #         'use_cbar': False,
    #         'use_rho_bar': False,
    #         'use_h_bar': False,
    #         'scaleBeta': False,
    #         'K': 1.0,
            
    #         'thermalConductivity' : 0.0,
    #     },
    #     'diffusionSwitch':{
    #         'scheme': ViscositySwitch.NoneSwitch,
    #         'limitXi': False,
    #     },
    #     'domain': config.domain,
    #     'kernel': KernelType.B7,
    #     'verbose': True,
    #     'gamma': compParams.gamma,
    #     'fluid':{
    #         'gamma': compParams.gamma,
    #     },
    #     'support':{
    #         'iterations': 1,
    #         'LUT': lut
    #     }
    # }

    # dsphConfig['support'] = {
    #     'LUT' : lut
    # }
    # dsphConfig['energyScheme'] = EnergySchemeDiffSPH.CRK
    # dsphConfig['targetNeighbors'] = config.targetNeighbors
    # device = currentState.positions.device

    # with TimedBlock("state copy", use_cuda=True, device=device):
    #     diffSPHState = CompState(
    #         positions = currentState.positions,
    #         velocities = currentState.velocities,
    #         densities = currentState.densities,
    #         supports = currentState.supports,
    #         internalEnergies = currentState.internalEnergies,
    #         totalEnergies = currentState.totalEnergies,
    #         entropies = currentState.entropies,
    #         soundspeeds= currentState.soundspeeds,
    #         masses = currentState.masses,
    #         kinds = currentState.kinds,
    #         materials = currentState.materials,
    #         UIDs = currentState.UIDs,
    #         pressures = currentState.pressures,
    #         omega = torch.ones_like(currentState.densities),
    #         alphas = currentState.alphas,
    #         alpha0s = currentState.alpha0s,
    #     )

    # wrappedKernel = getSPHKernelv2(KernelType.B7)
    # verletScale = 1
    # t = currentSystem.t
    # diffSPHState = enforceDirichlet(diffSPHState, dsphConfig, t, dt)    
    
    # with TimedBlock('evaluate support', use_cuda=True, device=device):
    #     # rho_optimal, h_optimal, _, *_ = evaluateOptimalSupport(currentState, config, compParams, SupportScheme.Gather, None)
    #     # currentState.supports = h_optimal
    #     # currentState.densities = rho_optimal
    #     rho, h_i_new, rhos, hs, neighborhood = evaluateOptimalSupportOwen(diffSPHState, dsphConfig['kernel'], None, SupportScheme.Gather, dsphConfig)

    #     diffSPHState.supports = h_i_new

    # with TimedBlock('evaluate neighborhood', use_cuda=True, device=device):
    #     neighborhood, neighbors = evaluateNeighborhood(diffSPHState, config.domain, KernelType.B7, verletScale = 1.0, mode = SupportSchemeDiffSPH.SuperSymmetric, priorNeighborhood=None)

    # with TimedBlock('compute density', use_cuda=True, device=device):
    #     diffSPHState.densities = computeDensity(diffSPHState, wrappedKernel, neighbors.get('noghost'), SupportSchemeDiffSPH.Gather, config)

    # # with record_function("[deltaSPH] - 05 - Dirichlet BC"):
    # diffSPHState = enforceDirichlet(diffSPHState, dsphConfig, t, dt)    

    # # with TimedBlock('enforce dirichlet', use_cuda=True, device=device):
    #     # enforceDirichlet(diffSPHState, t, dt, config, compParams)
    
    # with TimedBlock('compute EOS', use_cuda=True, device=device):
    #     currentState.entropies, _, currentState.pressures, currentState.soundspeeds = idealGasEOS(
    #         A = None,
    #         u = currentState.internalEnergies,
    #         P = None,
    #         rho = currentState.densities,
    #         gamma = compParams.gamma,
    #     )

    # with TimedBlock('compute acceleration', use_cuda=True, device=device):
    #     dvdt, ap_ij_dsph, av_ij_dsph = compSPH_acceleration(diffSPHState, wrappedKernel, neighbors.get('noghost'), SupportSchemeDiffSPH.SuperSymmetric, dsphConfig)

    # with TimedBlock('compute dudt', use_cuda=True, device=device):
    #     dudt = compSPH_dudt(diffSPHState, wrappedKernel, neighbors.get('noghost'), SupportSchemeDiffSPH.SuperSymmetric, dsphConfig)

    # with TimedBlock('compute balance term', use_cuda=True, device=device):
    #     v_halfstep = diffSPHState.velocities + 0.5 * config.dt * dvdt

    #     currentState.f_ij = compute_fij(diffSPHState, wrappedKernel, neighbors.get('noghost'), SupportScheme.SuperSymmetric, dsphConfig, config.dt, v_halfstep, ap_ij_dsph, av_ij_dsph)
    #     currentState.ap_ij = ap_ij_dsph
    #     currentState.av_ij = av_ij_dsph

    # with TimedBlock('compute warp', use_cuda=True, device=device):
    t = currentSystem.t
    rho_optimal, h_optimal, currentSystem.adjacency, *_ = evaluateOptimalSupport(currentState, config, compParams, SupportScheme.Gather, currentSystem.adjacency)
    currentState.supports = h_optimal
    currentState.densities = rho_optimal

    verletScale = config.verletScale

    adjacency = buildVerletList(
        currentState, 
        config.domain, verletScale = verletScale, supportMode = SupportScheme.SuperSymmetric,
        priorNeighborhood = currentSystem.adjacency,
        verbose = False)
    currentSystem.adjacency = adjacency

    # with TimedBlock('compute csr', use_cuda=True, device=device):
        # csr_neighrs = coo_to_csr(neighbors.get('noghost')[0])
        # adjacency.i = neighbors.neighbors.row
        # adjacency.j = neighbors.neighbors.col
        # adjacency.numNeighbors = csr_neighrs.rowEntries.to(torch.int32)
        # adjacency.edgeOffsets = csr_neighrs.indptr.to(torch.int32)
        # # currentSystem.adjacency = adjacency

    currentState.densities = warpOperation(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Density,
            supportMode = SupportScheme.Gather, # cullen switch E.1 in the CRK paper uses gather for density estimation
        ),
        domain = config.domain,
        adjacency = adjacency,
    )
    if currentState.divergence is None:
        print('Warning: divergence is None, computing for the first time')
        drhodt = computeMomentumConsistent(
            currentState,
            config,
            supportScheme = SupportScheme.Gather, 
            adjacency = adjacency,
            gradH = gradHState
        )
        currentState.divergence = drhodt

    # enforceDirichlet(currentSystem.state, dsphConfig, t, dt)

    enforceDirichlet(currentSystem, t, dt, config, compParams)
    currentState.entropies, _, currentState.pressures, currentState.soundspeeds = idealGasEOS(
        A = None,
        u = currentState.internalEnergies,
        P = None,
        rho = currentState.densities,
        gamma = compParams.gamma,
    )

    if compParams.adaptiveSupportCorrections:
        omega = computeOmega(currentState, 
                OperationProperties(
                    kernel = config.kernel,
                    supportMode = SupportScheme.Gather, # E.5
                ),
                domain = config.domain,
                adjacency = adjacency
        )

        gradHState = GradHState(
            queryOmegas = omega
        )
    else:
        gradHState = None

    currentState.alphas, switchState = computeViscositySwitchTerms(
        dt,
        currentState, 
        config, compParams, 
        SupportScheme.SuperSymmetric, 
        adjacency)   


    dvdt, currentState.ap_ij, currentState.av_ij = computeCompSPHAccelWarp(
        queryParticles = currentState,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode =  SupportScheme.KernelMeanSymmetric
        ),
        domain = config.domain,
        conductivityParams= compParams.diffusionParams,

        queryEnergies = currentState.internalEnergies,
        queryVelocities= currentState.velocities,
        queryCs = currentState.soundspeeds,
        queryAlphas = currentState.alphas,
        queryPressures = currentState.pressures,

        adjacency = adjacency,
        gradHState = gradHState
    )

    dudt = computeCompSPHdudtWarp(
        queryParticles = currentState,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = SupportScheme.Gather #E.3
         ),
        domain = config.domain,
        conductivityParams= compParams.diffusionParams,

        queryEnergies = currentState.internalEnergies,
        queryVelocities= currentState.velocities,
        queryCs = currentState.soundspeeds,
        queryAlphas = currentState.alphas,
        queryPressures = currentState.pressures,

        adjacency = adjacency,
        gradHState = gradHState
    )

    # particles.alpha0s, switchState = updateViscositySwitch(particles, wrappedKernel, neighbors.get('noghost'), SupportScheme.Gather, config, dt, dvdt, switchState)

    currentState.alpha0s, switchState = updateViscositySwitch(
        switchState,
        dt, dvdt,
        currentState, 
        config, compParams, 
        SupportScheme.SuperSymmetric, 
        adjacency)   


    drhodt = computeMomentumConsistent(
        currentState,
        config,
        supportScheme = SupportScheme.Gather,
        adjacency = adjacency,
        gradH = gradHState
    )
    currentState.divergence = -drhodt / currentState.densities
    dEdt = currentState.masses * torch.einsum('ij,ij->i', currentState.velocities, (dvdt)) + currentState.masses * (dudt)

    # drhodt = torch.zeros_like(currentState.densities)
    # dEdt = torch.zeros_like(currentState.densities)

    # with TimedBlock('compute forcing', use_cuda=True, device=device):
    forcing = computeForcing(currentSystem, dt, t, config, compParams)
    dvdt += forcing / currentState.masses.view(-1,1)

    update = CompressibleSystemUpdate(
        dxdt = currentState.velocities.clone(),
        dvdt = dvdt,
        dudt = dudt,
        drhodt = drhodt,
        dEdt = dEdt,
        passive = torch.zeros(currentState.densities.shape, device=currentState.densities.device, dtype=torch.bool)
    )
    # update = enforceDirichletUpdate(update, diffSPHState, dsphConfig, t, dt)

    # with TimedBlock('enforce updates', use_cuda=True, device=device):
    enforceUpdates(update, currentSystem, dt, t, config, compParams)

    v_halfstep = currentState.velocities + 0.5 * dt * update.dvdt

    currentState.f_ij = computeCompSPHBalanceTermWarp(
        queryParticles = currentState,
        operationProperties = OperationProperties(
            kernel = config.kernel,
            supportMode = config.supportMode
        ),
        domain = config.domain,

        queryEnergies = currentState.internalEnergies,
        queryVelocities= v_halfstep,
        queryPressures = currentState.pressures,

        pairWise_pressureAccel= currentState.ap_ij,
        pairWise_viscosityAccel = currentState.av_ij,
        energyScheme = compParams.energyScheme,
        dt= dt.detach().cpu().item() if isinstance(dt, torch.Tensor) else dt,
        gamma = compParams.gamma,

        adjacency = adjacency,
        gradHState = gradHState
    )

    return update, adjacency, currentState