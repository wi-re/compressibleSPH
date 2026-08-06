from warpSPH.utils.support import volumeToSupportHelper

from ...systems.weaklyCompressible import WeaklyCompressibleState, WeaklyCompressibleSystem, WeaklyCompressibleSystemUpdate
from ...configurations.weaklyCompressible import WeaklyCompressibleSPHConfig
from ...configurations.simulationConfig import SimulationConfig
from typing import Any, Optional
from sphWarpCore import *
from ...modules.eos import idealGasEOS
import torch
import warp as wp

from .compressible import computeTimestep as computeTimestepCompressible
from .weaklyCompressible import computeTimestep as computeTimestepWeaklyCompressible


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