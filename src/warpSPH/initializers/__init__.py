"""Builds the initial `SimulationState`/`SimulationSystem` for a scheme from
its sampled regions and config; `initializeWeaklyCompressibleSimulation` is
the weakly-compressible entry point that `cases/weaklyCompressible.py` and
`cases/dambreak.py` call after regions have been sampled.
"""

from .weaklyCompressible import initializeSimulation as initializeWeaklyCompressibleSimulation

__all__ = [
    'initializeWeaklyCompressibleSimulation',
]