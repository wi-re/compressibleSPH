"""computeTimestep dispatcher.

Picks the weakly-compressible or (fully) compressible adaptive-timestep
implementation based on `isinstance(system, WeaklyCompressibleSystem)`; this
is the single entry point re-exported as `warpSPH.modules.timestep.computeTimestep`.
"""

from warpSPH.utils.support import volumeToSupport

from ...systems.weaklyCompressible import WeaklyCompressibleState, WeaklyCompressibleSystem, WeaklyCompressibleSystemUpdate
from ...configurations.weaklyCompressible import WeaklyCompressibleSPHConfig
from ...configurations.simulationConfig import SimulationConfig
from typing import Any, Optional
from warpSPHCore import *
from ...modules.eos import idealGasEOS
import torch
import warp as wp

from .compressible import computeTimestep as computeTimestepCompressible
from .weaklyCompressible import computeTimestep as computeTimestepWeaklyCompressible

__all__ = ['computeTimestep']


def computeTimestep(
    system: Any,
    config: SimulationConfig,
    compParams: Any,
    dt: Optional[float] = None,
    systemUpdate: Optional[WeaklyCompressibleSystemUpdate] = None,
):
    if isinstance(system, WeaklyCompressibleSystem):
        return computeTimestepWeaklyCompressible(
            system = system,
            config = config,
            compParams = compParams,
            dt = dt,
            systemUpdate = systemUpdate
        )
    else:
        return computeTimestepCompressible(
            system = system,
            config = config,
            compParams = compParams,
            dt = dt,
        )