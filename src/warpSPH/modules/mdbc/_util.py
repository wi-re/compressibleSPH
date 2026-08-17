"""Shared helper for the mDBC entry points' early-exit guard."""

import torch

__all__ = ['stateHasBoundaryParticles']


def stateHasBoundaryParticles(currentState, config) -> bool:
    """Whether this run has any kind==1 (boundary) particles.

    `kinds` is a run-constant field: assigned once at particle generation and
    never mutated afterward (see `initializers/weaklyCompressible.py`), so its
    boundary/no-boundary pattern only needs a device readback once per run,
    not once per step. The result is cached on `config`, which -- unlike
    `currentState` (recreated per RK stage/step by `initializeNewState()`,
    per `warpSPHIntegrators.fields._state_initialize`) -- is the same object
    for the whole run: the runner mutates it in place (e.g. `ctx.config.dt`)
    rather than replacing it.
    """
    cached = getattr(config, '_hasBoundaryParticles', None)
    if cached is None:
        cached = bool(torch.any(currentState.kinds == 1).item())
        config._hasBoundaryParticles = cached
    return cached
