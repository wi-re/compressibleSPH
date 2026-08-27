"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 4, 2026-08-26): is DFSPH's bulk
density error (documented in Part 2 as a wall-adjacent-vs-bulk *magnitude*,
`probe_dfsphWallDensityProfile.py`) a systematic *bias*, or zero-mean noise?
Answers this directly on `kolmogorovIncompressible` (periodic, no boundary,
no free surface) since that removes any wall/mDBC involvement from the
question. Feeds directly into `probe_incompressibleGaugeDrift.py`'s finding
that `solveIncompressible`'s `sourceTerm = rho0 - rhoStar` has a persistent
negative mean at (essentially) every step -- if the density error is signed
rather than noisy, that's the mechanism.

Finding (nx=64, 300 steps): `mean(rho-1) = +2.71e-3` against
`mean|rho-1| = 3.54e-3` -- the signed bias accounts for ~76% of the unsigned
error, and 82% of particles sit above `rho0`. This is predominantly a
systematic bias, not noise, and it's the same sign/order of magnitude as the
`sourceTerm` bias driving `solveIncompressible`'s pressure-mean runaway --
strong evidence Part 2's still-open bulk-gap finding and Part 4's gauge-drift
finding share one upstream cause (DFSPH's bulk pressure projection running
systematically dense).

Usage: `python scripts/probe_densitySign.py [--nx 64] [--nsteps 300]`
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=64)
parser.add_argument('--nsteps', type=int, default=300)
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

from warpSPH.cases import kolmogorovIncompressible as mod
from warpSPH.runner.cli import caseMain

captured = {}
_orig = mod.diagnostics


def _wrapped(ctx, state):
    captured['state'] = state
    return _orig(ctx, state)


mod.diagnostics = _wrapped
mod.kolmogorovIncompressibleCase.diagnostics = _wrapped

result = caseMain(mod.kolmogorovIncompressibleCase, argv=[
    '--nx', str(args.nx), '--nSteps', str(args.nsteps), '--tLimit', '1000.0',
    '--quiet', '--no-store', '--no-plot',
])

state = captured['state'].state
rho = state.densities
dev = rho - 1.0
print(f"n particles: {rho.shape[0]}")
print(f"mean(rho-1)   = {dev.mean().item(): .6e}   (signed bulk bias)")
print(f"mean|rho-1|   = {dev.abs().mean().item(): .6e}   (unsigned error)")
print(f"std(rho)      = {rho.std().item(): .6e}")
print(f"fraction rho>1: {(rho > 1).float().mean().item():.4f}")
print(f"fraction rho<1: {(rho < 1).float().mean().item():.4f}")
