"""Compatibility layer for weakly-compressible example helpers.

The source-of-truth implementations now live in warpSPH:
- case utilities: warpSPH.caseUtils
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
]
