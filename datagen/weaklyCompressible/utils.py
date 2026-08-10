"""Compatibility layer for weakly-compressible datagen helpers.

The source-of-truth implementations now live in warpSPH:
- case utilities: warpSPH.caseUtils
- HDF5 config serialization helpers: warpSPH.io
"""

from warpSPH.caseUtils import (
    SimulationProperties,
    buildPresetObstacles,
    buildObstacleSDF,
    build_sdfs,
    buildDomain,
    buildRegions,
    sampleNoise,
    setupFreestream,
    setupKolmogorov,
)
from warpSPH.io.io import copy_dict_to_h5, restore_config_from_h5


# Backward-compatible legacy name.
def restoreConfig_from_h5(group, indent=0):
    return restore_config_from_h5(group, indent=indent)


__all__ = [
    "SimulationProperties",
    "buildPresetObstacles",
    "buildObstacleSDF",
    "build_sdfs",
    "buildDomain",
    "buildRegions",
    "sampleNoise",
    "setupFreestream",
    "setupKolmogorov",
    "copy_dict_to_h5",
    "restore_config_from_h5",
    "restoreConfig_from_h5",
]
