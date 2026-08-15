"""Support types for `gas.py`'s ideal-gas EOS path: which thermodynamic
quantity to treat as the source (`EOSSource`) and a minimal per-fluid
property bundle (`fluidProperties`).

Note: this `fluidProperties` is a separate, smaller dataclass from
`warpSPH.configurations.moduleConfigurations.fluidProperties.fluidProperties`
(the one actually referenced as `schemeConfig.fluid` elsewhere in the
codebase) — only `gas.py` uses this module's version.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

__all__ = ['EOSSource', 'fluidProperties']

class EOSSource(Enum):
    internalEnergy = 1
    pressure = 2
    specificEntropy = 3
    soundSpeed = 4


@dataclass
class fluidProperties:
    restDensity: float = field(metadata={"description": "Rest density of the fluid"})

    gamma: Optional[float] = field(default=None, metadata={"description": "Adiabatic index"})
    polytropicIndex: Optional[float] = field(default=None, metadata={"description": "Polytropic index"})
    kappa: Optional[float] = field(default=None, metadata={"description": "Kappa"})

    fixedSoundSpeed: Optional[float] = field(default=None, metadata={"description": "Fixed sound speed"})

