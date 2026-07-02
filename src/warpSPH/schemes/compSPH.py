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

from ..modules.shockCapturing.CullenHopkins import computeHopkinsTerms, computeHopkinsUpdate

lut = None

from sphWarpCore import *

import time
import torch

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
    # currentSystem.adjacency = None

    t = currentSystem.t
    rho_optimal, h_optimal, currentSystem.adjacency, *_ = evaluateOptimalSupport(currentState, config, compParams, SupportScheme.Gather, currentSystem.adjacency)
    currentState.supports = h_optimal
    currentState.densities = rho_optimal

    verletScale = config.verletScale

    adjacency = buildVerletList(
        currentState, 
        config.domain, verletScale = verletScale, supportMode = SupportScheme.SuperSymmetric,
        priorNeighborhood = None,
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