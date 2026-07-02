from dataclasses import dataclass
from typing import Optional
import torch

@dataclass(slots = True)
class ViscositySwitchState:
    alpha0s: torch.Tensor            # Alpha0 values that get integrated over time
    alphas:  torch.Tensor            # The alpha values for the Cullen-Dehnen viscosity model
    M:       Optional[torch.Tensor]  # Gradient Correction matrix from CRKSPH
    M_inv:   Optional[torch.Tensor]  # Correction is performed using the inverse matrix
    div:     Optional[torch.Tensor]  # Divergence of the velocity field
    ddivdt:  Optional[torch.Tensor]  # Second order divergence of the velocity field
    Shear:   Optional[torch.Tensor]  # Shear tensor
    Rot:     Optional[torch.Tensor]  # Rotation tensor
    R:       Optional[torch.Tensor]  # R term from Cullen and Dehnen 2010
    Xi:      Optional[torch.Tensor]  # Limiter Term, unnamed in CRKSPH
    v_sig:   Optional[torch.Tensor]  # Signal Velocity Term


