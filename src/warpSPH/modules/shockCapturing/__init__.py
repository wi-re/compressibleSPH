"""Time-varying artificial-viscosity switches (Cullen & Dehnen 2010, Hopkins' modified form).

Dispatches between the two per-particle alpha-viscosity schemes on
``schemeConfig.viscositySwitchParams.scheme``, plus their shared support code
(shear/rotation tensor, CRK-style gradient correction matrix, signal velocity).
"""

from .wrapper import computeViscositySwitchTerms, updateViscositySwitch
from .switchState import ViscositySwitchState

__all__ = ['computeViscositySwitchTerms', 'updateViscositySwitch', 'ViscositySwitchState']