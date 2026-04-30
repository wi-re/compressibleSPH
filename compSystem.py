from integrators import *
from dataclasses import dataclass
import torch
from typing import Optional
from sphWarpCore import *

@dataclass
class CompressibleState(BaseState):
    positions : torch.Tensor = integrated('dxdt', tags=('position',))
    velocities: torch.Tensor = integrated('dvdt', tags=('velocity',))
    supports : torch.Tensor = constant(tags=('particle_support',))
    masses : torch.Tensor = constant(tags=('particle_mass',))
    densities : torch.Tensor = constant(tags=('particle_density',))

    kinds : torch.Tensor = constant(tags=('particle_kind',))
    materials : torch.Tensor = constant(tags=('particle_material',))
    UIDs : torch.Tensor = constant(tags=('particle_UID',))
    UIDcounter : int = constant(tags=('particle_UIDcounter',))

    internalEnergies : torch.Tensor = integrated('dudt', tags=('internalEnergy',))
    totalEnergies : torch.Tensor = constant(tags=('energy',))
    entropies : torch.Tensor = constant(tags=('soundSpeed',))
    pressures : torch.Tensor = constant(tags=('damping',))
    soundspeeds : torch.Tensor = constant(tags=('soundSpeed',))

    divergence : torch.Tensor = constant(tags=('velocity_divergence',))
    alpha0s: torch.Tensor = constant(tags=('alpha0',))
    alphas: torch.Tensor = constant(tags=('alpha',))

@dataclass
class CompressibleSystemUpdate:
    dxdt: torch.Tensor = tagged(tags=('position_derivative',))
    dvdt: torch.Tensor = tagged(tags=('velocity_derivative',))
    dudt: torch.Tensor = tagged(tags=('internalEnergy_derivative',))

@dataclass
class CompressibleSystem(BaseIntegrationSystem):
    state: CompressibleState = reference_state(tags=('physics_state',))
    adjacency: Optional[AdjacencyList] = None
    domain: Optional[DomainDescription] = None
    t: float = 0.0
    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        verbosePrint(verbose, f'Initializing new state [t={self.t}]')
        return CompressibleSystem(state=state.initializeNewState(), adjacency=self.adjacency, t=self.t, domain=self.domain)
    
    def apply_position_update(self, update, spec: PositionUpdateSpec, **kwargs):
        return update_position(self, update, spec, 'position', 'position_derivative', 'velocity', 'velocity_derivative')
    def apply_velocity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'velocity', 'velocity_derivative')
    def apply_quantity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'internalEnergy', 'internalEnergy_derivative')
    def apply_state_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        # Note: DO NOT advance self.t here. Time is managed by the integrator.
        position_spec = PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend)
        self.apply_position_update(update, position_spec, **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        self.apply_quantity_update(update, spec, **kwargs)
        return self
    
    def finalize(self, initialState, dt, returnValues, updateValues, weights = ..., *args, **kwargs):
        self.adjacency = returnValues[-1][0]  # Assuming the adjacency list is the last return value from the derivative function
        # Copy the last substeps values into the current state to ensure the final state is correct
        lastState = returnValues[-1][1]  # Assuming the state is the second return value from the derivative function
        # General attributes
        self.state.supports.copy_(lastState.supports)
        self.state.densities.copy_(lastState.densities)

        # Compressible system specific attributes (internal eneryg is integrated)
        self.state.totalEnergies.copy_(lastState.totalEnergies)
        self.state.entropies.copy_(lastState.entropies)
        self.state.pressures.copy_(lastState.pressures)
        self.state.soundspeeds.copy_(lastState.soundspeeds)

        # Information for artificial viscosity switches
        self.state.divergence.copy_(lastState.divergence)
        self.state.alpha0s.copy_(lastState.alpha0s)
        self.state.alphas.copy_(lastState.alphas)

        return super().finalize(initialState, dt, returnValues, updateValues, weights, *args, **kwargs)
    