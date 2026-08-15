"""Compatibility layer for weakly-compressible example helpers.

The source-of-truth implementations now live in warpSPH:
- case utilities: warpSPH.caseUtils

Only one file still imports this: ``13-open-flow.ipynb``, the last notebook in
this family that has not been ported to the current style (see
MIGRATION_PLAN.md). Everything else imports from ``warpSPH.caseUtils``
directly, and new code should too. When that notebook is ported, this module
goes with it -- it exists solely so its bare ``from utils import ...`` keeps
resolving until then.
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
