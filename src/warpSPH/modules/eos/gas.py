
from typing import Any, List, Union, Optional
import torch
from ...systems import CompressibleState
from .props import fluidProperties, EOSSource

def computeQuantitiesIdealGas(
    fluidProps: fluidProperties,
    rho: torch.Tensor,
    u: Optional[torch.Tensor] = None,
    P: Optional[torch.Tensor] = None,
    c_s: Optional[torch.Tensor] = None,
    A: Optional[torch.Tensor] = None,
    EOSSource: EOSSource = EOSSource.internalEnergy,
):
    gamma = fluidProps.gamma

    P_, u_, A_, rho_ = P, u, A, rho
    
    if EOSSource == EOSSource.internalEnergy and u is not None:
        P_ = (gamma - 1) * rho * u
        A_ = u * (gamma - 1) * rho**(1 - gamma)
        c_s = torch.sqrt(u.abs() * gamma * (gamma - 1))
    elif EOSSource == EOSSource.pressure and P is not None:
        u_ = P / rho / (gamma - 1)
        A_ = P / rho**gamma
        c_s = torch.sqrt(gamma * P.abs() / rho)
    elif EOSSource == EOSSource.specificEntropy and A is not None:
        u_ = A * rho**(gamma - 1) / (gamma - 1)
        P_ = A * rho**gamma
        c_s = torch.sqrt(gamma * rho ** (gamma - 1) * A)
    elif EOSSource == EOSSource.soundSpeed and c_s is not None:
        P_ = c_s**2 * rho / gamma
        u_ = P_ / rho / (gamma - 1)
        A_ = P_ / rho**gamma
    else:
        raise ValueError(f"Invalid EOSSource or missing required parameter for EOSSource: {EOSSource}")
    
    return A_, u_, P_, c_s



def idealGas(
    particles: CompressibleState,
    fluidProps: Union[fluidProperties, List[fluidProperties]],
    EOSSource: EOSSource = EOSSource.internalEnergy,
):
    """
    Compute the equation of state for an ideal gas.

    Parameters:
    - particles: The particle data structure containing properties like density, internal energy, etc.
    - fluidProps: Either a single fluidProperties object or a list of them for different fluids.
    - EOSSource: The source from which to compute the equation of state (internal energy, pressure, etc.).

    Returns:
    - Updated particle properties based on the ideal gas equation of state.
    """
    # Implementation of the ideal gas EOS computation goes here
    
    if isinstance(fluidProps, list):
        numMaterials = len(fluidProps)
        for i in range(numMaterials):
            # Compute EOS for each material based on the specified EOSSource
            mask = (particles.materials == i)
            rhos = particles.densities[mask]
            us = particles.internalEnergies[mask] if hasattr(particles, 'internalEnergies') and EOSSource == EOSSource.internalEnergy else None
            Ps = particles.pressures[mask] if hasattr(particles, 'pressures') and EOSSource == EOSSource.pressure else None
            cs = particles.soundspeeds[mask] if hasattr(particles, 'soundspeeds') and EOSSource == EOSSource.soundSpeed else None
            if cs is None and fluidProps[i].fixedSoundSpeed is not None:
                cs = torch.full_like(rhos, fluidProps[i].fixedSoundSpeed)
            As = particles.entropies[mask] if hasattr(particles, 'entropies') and EOSSource == EOSSource.specificEntropy else None

            A_, u_, P_, c_s_ = computeQuantitiesIdealGas(fluidProps[i], rhos, us, Ps, cs, As, EOSSource)

            # Update the particle properties
            if hasattr(particles, 'internalEnergies'):
                particles.internalEnergies[mask] = u_
            if hasattr(particles, 'pressures'):
                particles.pressures[mask] = P_
            if hasattr(particles, 'soundspeeds'):
                particles.soundspeeds[mask] = c_s_
            if hasattr(particles, 'entropies'):
                particles.entropies[mask] = A_
    else:
        # Single fluid case
        rhos = particles.densities
        us = particles.internalEnergies if hasattr(particles, 'internalEnergies') and EOSSource == EOSSource.internalEnergy else None
        Ps = particles.pressures if hasattr(particles, 'pressures') and EOSSource == EOSSource.pressure else None
        cs = particles.soundspeeds if hasattr(particles, 'soundspeeds') and EOSSource == EOSSource.soundSpeed else None
        if cs is None and fluidProps.fixedSoundSpeed is not None:
            cs = torch.full_like(rhos, fluidProps.fixedSoundSpeed)
        As = particles.entropies if hasattr(particles, 'entropies') and EOSSource == EOSSource.specificEntropy else None

        A_, u_, P_, c_s_ = computeQuantitiesIdealGas(fluidProps, rhos, us, Ps, cs, As, EOSSource)

        # Update the particle properties
        if hasattr(particles, 'internalEnergies'):
            particles.internalEnergies = u_
        if hasattr(particles, 'pressures'):
            particles.pressures = P_
        if hasattr(particles, 'soundspeeds'):
            particles.soundspeeds = c_s_
        if hasattr(particles, 'entropies'):
            particles.entropies = A_