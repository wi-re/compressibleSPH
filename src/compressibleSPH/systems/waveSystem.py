import torch
from typing import Optional
from dataclasses import dataclass

from ..config import SimulationConfig
from ..sampling import sampleParticles

from ..utils import *
from integrators import *
from sphWarpCore import *


@dataclass
class WaveSystemStatev3(BaseState):
    positions : torch.Tensor = constant(tags=('particle_position',))
    supports : torch.Tensor = constant(tags=('particle_support',))
    masses : torch.Tensor = constant(tags=('particle_mass',))
    densities : torch.Tensor = constant(tags=('particle_density',))

    kinds : torch.Tensor = constant(tags=('particle_kind',))
    materials : torch.Tensor = constant(tags=('particle_material',))
    UIDs : torch.Tensor = constant(tags=('particle_UID',))
    UIDcounter : int = constant(tags=('particle_UIDcounter',))

    u : torch.Tensor = integrated('dxdt', tags=('position',))
    v : torch.Tensor = integrated('dvdt', tags=('velocity',))
    c : torch.Tensor = constant(tags=('soundSpeed',))
    damping : torch.Tensor = constant(tags=('damping',))

@dataclass
class WaveSystemUpdatev3:
    dudt : torch.Tensor = tagged(tags=('position_derivative',))
    dvdt : torch.Tensor = tagged(tags=('velocity_derivative',))

@dataclass
class WaveSystemv3(BaseIntegrationSystem):
    state: WaveSystemStatev3 = reference_state(tags=('oscillator_state',))
    adjacency: Optional[AdjacencyList] = None
    domain: Optional[DomainDescription] = None
    t: float = 0.0
    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        verbosePrint(verbose, f'Initializing new state [t={self.t}]')
        return WaveSystemv3(state=state.initializeNewState(), adjacency=self.adjacency, t=self.t, domain=self.domain)
    
    def apply_position_update(self, update, spec: PositionUpdateSpec, **kwargs):
        return update_position(self, update, spec, 'position', 'position_derivative', 'velocity', 'velocity_derivative')
    def apply_velocity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'velocity', 'velocity_derivative')
    def apply_quantity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return self
    def apply_state_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        # Note: DO NOT advance self.t here. Time is managed by the integrator.
        position_spec = PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend)
        self.apply_position_update(update, position_spec, **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        self.apply_quantity_update(update, spec, **kwargs)
        return self
    
    def finalize(self, initialState, dt, returnValues, updateValues, weights = ..., *args, **kwargs):
        self.adjacency = returnValues[-1][0]  # Assuming the adjacency list is the last return value from the derivative function
        return super().finalize(initialState, dt, returnValues, updateValues, weights, *args, **kwargs)
    

from ..config import WaveCaseConfig

def sampleInitialWaveState(nx, config: SimulationConfig, caseConfig: WaveCaseConfig):
    particles = sampleParticles(nx, config)
    particleState = WaveSystemStatev3(
        positions=particles.positions,
        supports=particles.supports,
        masses=particles.masses,
        densities=particles.densities,
        
        kinds=torch.zeros(particles.positions.shape[0], device=config.device, dtype=torch.int32),
        materials=torch.zeros(particles.positions.shape[0], device=config.device, dtype=torch.int32),
        UIDs=torch.arange(particles.positions.shape[0], device=config.device, dtype=torch.int32),
        UIDcounter=particles.positions.shape[0],
        
        u=torch.zeros(particles.positions.shape[0], device=config.device),
        v=torch.zeros(particles.positions.shape[0], device=config.device),
        c=torch.ones(particles.positions.shape[0], device=config.device) * caseConfig.defaultSpeed,
        damping=torch.zeros(particles.positions.shape[0], device=config.device)
    )
    return particleState


from typing import List
def computeDt(waveSystem: WaveSystemv3, config: SimulationConfig, caseConfig: WaveCaseConfig, args, obstacleSpeeds: List[float], verbose=False):
    # Compute CFL number based on initial conditions
    n = waveSystem.state.u.shape[0]
    nx = int(n**0.5)
    # print(f'Assuming square grid with nx={nx}, ny={nx}')
    domainArea = (config.domain.max[0] - config.domain.min[0]) * (config.domain.max[1] - config.domain.min[1])
    # print(f'Particle area: {domainArea/n}, particle spacing: {(domainArea/n)**0.5}')
    dx = (domainArea/n)**0.5

    dt = 0.02
    cflNumber = max(obstacleSpeeds + [caseConfig.defaultSpeed]) * dt / dx

    # runningState = waveSystem.initializeNewState(verbose=verbose)
    hMax = waveSystem.state.supports.max().item()
    if verbose:
        print(f'Max wave speed: {waveSystem.state.c.max().item():.4f}')
        print(f'Max support radius: {hMax:.4f}')
        print(f'CFL number (based on dx): {cflNumber:.4f} [dx={dx:.4f}, dt={dt:.4f}]')
        print(f'CFL number (based on hMax): {max(obstacleSpeeds + [caseConfig.defaultSpeed]) * dt / hMax:.4f} [hMax={hMax:.4f}, dt={dt:.4f}]')

    targetCFL = config.cflFactor
    # Adjust dt such that the CFL number based on hMax is equal to targetCFL
    dt = targetCFL * hMax / max(obstacleSpeeds + [caseConfig.defaultSpeed])
    if verbose:
        print(f'Adjusted dt for target CFL of {targetCFL}: {dt:.4f}')
        print(f'New CFL number (based on dx): {max(obstacleSpeeds + [caseConfig.defaultSpeed]) * dt / dx:.4f}')
        print(f'New CFL number (based on hMax): {max(obstacleSpeeds + [caseConfig.defaultSpeed]) * dt / hMax:.4f}')
    return dt