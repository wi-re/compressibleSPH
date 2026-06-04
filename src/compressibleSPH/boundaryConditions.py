from dataclasses import dataclass
from enum import Enum
from typing import Callable, Tuple, Dict, Any
from compressibleSPH.utils.sdf import *
import torch

class BoundaryConditionType(Enum):
    constant = "constant"
    dynamic = "dynamic"
    reflective = "reflective"
    extending = "extending"
    forcing = "forcing"
    dynamic_forcing = "dynamic_forcing"

class VectorProjectionType(Enum):
    none = "none"
    normal = "normal"
    tangential = "tangential"

@dataclass
class BoundaryCondition:
    type: BoundaryConditionType
    sdf: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]] # SDF function defining the boundary returns distance and normal

    # The functions take as input:
    # - The current particle state (as an Any type, which can be a custom dataclass or a dictionary containing all relevant particle properties)
    # - The simulation configuration (as an Any type, which can be a custom dataclass or a dictionary containing all relevant simulation parameters)
    # - The scheme configuration (as an Any type, which can be a custom dataclass or a dictionary containing all relevant scheme parameters)
    # - The particle positions (torch.Tensor of shape [N, dim])
    # - The distance to the boundary (torch.Tensor of shape [N])
    # - The normal to the boundary (torch.Tensor of shape [N, dim])
    # - The current time (torch.Tensor scalar)
    # - The current time step (torch.Tensor scalar)
    # The function returns
    # - The new value of the quantity being updated
    # This function needs to be defined for each quantity that is updated by the boundary condition (e.g. density, velocity, energy, etc.)
    dirichletFunctions: Dict[str, Callable[[Any, Any, Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None
    # Update functions work the same as dirichlet functions but instead target the update of the quantity instead of its value. This is useful for dynamic boundary conditions where the update of the quantity depends on its current value and the state of the system.
    updateFunctions: Dict[str, Callable[[Any, Any, Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None
    # Forcing functions work the same but return the forcing term. This gives a more convenient wrapper for some types of boundary conditions (e.g. dynamic_forcing) where the boundary condition is more naturally expressed as a forcing term rather than a direct update to the quantity.
    forcingFunctions: Dict[str, Callable[[Any, Any, Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None

    # Vector projection type for handling vector quantities at the boundary
    vectorProjectionType: VectorProjectionType = VectorProjectionType.none