
from sphWarpCore import *
# from ...systems.compSPH import CompSPHSystem, CompSPHState
from ...configurations.compSPHConfig import CompSPHConfig
from ...configurations.simulationConfig import SimulationConfig
from ...systems.compressibleMonaghan import CompressibleSystemUpdate
import torch

from ...systems.baseState import BaseState
from ...configurations import SimulationConfig, CompressibleSPHConfig
from typing import Any

def enforceDirichlet(
    system: Any,
    t: float,
    dt: float,
    config: SimulationConfig,
    compParams: CompSPHConfig,
):
    for bc in compParams.boundaryConditions:
        if bc.dirichletFunctions is not None and len(bc.dirichletFunctions) > 0:
            for varName, dirichletFn in bc.dirichletFunctions.items():
                #print(f'Enforcing dirichlet for {varName}')
                if hasattr(system, 'state'):
                    var = getattr(system.state, varName)
                    d, n = bc.sdf(system.state.positions)
                    updatedValues = dirichletFn(system.state, config, compParams, system.state.positions, d, n, t, dt)
                    var[d < 0] = updatedValues[d < 0]
                    # print(f'Enforced dirichlet for {varName}')
                else:
                    var = getattr(system, varName)
                    d, n = bc.sdf(system.positions)
                    updatedValues = dirichletFn(system, config, compParams, system.positions, d, n, t, dt)
                    var[d < 0] = updatedValues[d < 0]
                    # print(f'Enforced dirichlet for {varName}')
                
def computeForcing(
    system: Any,
    dt: float,
    t: float,
    config: SimulationConfig,
    compParams: CompSPHConfig,
):
    totalForcing = torch.zeros_like(system.state.positions) if hasattr(system, 'state') else torch.zeros_like(system.positions)
    for bc in compParams.boundaryConditions:
        if bc.forcingFunctions is not None and len(bc.forcingFunctions) > 0:
            for forcingFn in bc.forcingFunctions:
                if hasattr(system, 'state'):
                    d, n = bc.sdf(system.state.positions)
                    forcingValues = forcingFn(system.state, config, compParams, system.state.positions, d, n, t, dt)
                    totalForcing[d < 0] += forcingValues[d < 0]
                else:
                    d, n = bc.sdf(system.positions)
                    forcingValues = forcingFn(system, config, compParams, system.positions, d, n, t, dt)
                    totalForcing[d < 0] += forcingValues[d < 0]
    return totalForcing

def enforceUpdates(
    updates: CompressibleSystemUpdate,
    system: Any,
    dt: float,
    t: float,
    config: SimulationConfig,
    compParams: CompSPHConfig,
):
    for bc in compParams.boundaryConditions:
        if bc.updateFunctions is not None and len(bc.updateFunctions) > 0:
            for varName, updateFn in bc.updateFunctions.items():
                # print(f'Enforcing updates for {varName}')
                if hasattr(system, 'state'):
                    var = getattr(updates, varName)
                    d, n = bc.sdf(system.state.positions)
                    updatedValues = updateFn(system.state, config, compParams, system.state.positions, d, n, t, dt)
                    var[d < 0] = updatedValues[d < 0]
                else:
                    var = getattr(updates, varName)
                    d, n = bc.sdf(system.positions)
                    updatedValues = updateFn(system, config, compParams, system.positions, d, n, t, dt)
                    var[d < 0] = updatedValues[d < 0]
                # print(f'Enforced updates for {varName}')