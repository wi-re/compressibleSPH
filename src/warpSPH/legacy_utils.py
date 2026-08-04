"""Legacy compatibility module for dill-unpickled callables.

Older datasets may store callables with module path ``utils``. This module
provides the expected symbols under a stable package namespace so those
callables can still be restored when loading from outside datagen folders.
"""

from __future__ import annotations

from .caseUtils import (
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


def copy_dict_to_h5(*args, **kwargs):
    from .io import copy_dict_to_h5 as _copy_dict_to_h5

    return _copy_dict_to_h5(*args, **kwargs)


def restore_config_from_h5(*args, **kwargs):
    from .io import restore_config_from_h5 as _restore_config_from_h5

    return _restore_config_from_h5(*args, **kwargs)


def restoreConfig_from_h5(*args, **kwargs):
    return restore_config_from_h5(*args, **kwargs)


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
