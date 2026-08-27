"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 4, 2026-08-27): is DFSPH's
systematic *positive* bulk density bias (`probe_densitySign.py`:
`mean(rho-1) = +2.7e-3`, ~76% of the unsigned error, 82% of particles above
`rho0`) a **bug** somewhere in the momentum/pressure/shift kernels -- the
plan's option (1), "trace it to a specific asymmetric formula and fix it" --
or a **structural property of the SPH summation density itself**?

The test needs no dynamics at all, which is exactly what makes it decisive.
Take the case's own initial state (a perfect lattice, mass-normalised so
`mean(rho) == rho0` by construction), displace every particle by an
increasing amount of random jitter, and recompute the density. No solver,
no timestep, no kernel of this codebase's *scheme* involved -- only
`computeDensities`.

The prediction, if the bias is structural: `mean_i rho_i` must rise
monotonically with disorder and can never fall below the lattice value.
For equal masses, `sum_i rho_i = m * sum_{ij} W(x_i - x_j)`, which by
Parseval equals `m * sum_k What(k) |rhohat(k)|^2` with `What(k) >= 0` for a
positive-definite kernel (Wendland is one). A perfect lattice puts all its
spectral weight on reciprocal-lattice vectors, where the (bandwidth-limited)
`What` is ~0; *any* disorder moves weight to small `k` where `What` is
large, and every such contribution is non-negative. So the lattice
*minimises* the particle-averaged summation density, and disorder can only
raise it -- a one-sided, systematic, positive bias with no bug behind it.

Usage: `python scripts/probe_densityBiasVsDisorder.py [--nx 64]`
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=64)
parser.add_argument('--amplitudes', type=float, nargs='*',
                    default=[0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch

from warpSPH.cases import kolmogorovIncompressible as mod
from warpSPH.runner.cli import caseMain
from warpSPH.modules import computeDensities


class _Done(Exception):
    pass


_origBuild = mod.buildSystem


def _wrapped(ctx):
    system = _origBuild(ctx)
    state = system.state
    rho0 = ctx.param('rho0')
    x0 = state.positions.clone()
    # `dx` is the lattice spacing the case itself derives from the sampled
    # masses (`sample/weaklyCompressible.py`), so jitter is reported in units
    # of particle spacing, not domain length.
    dx = float(state.masses.min().item() ** (1.0 / ctx.config.dim))
    g = torch.Generator(device=x0.device)

    print(f"n={x0.shape[0]}  dx={dx:.5g}  h={state.supports.mean().item():.5g}")
    print(f"{'jitter/dx':>10} {'mean(rho-1)':>13} {'mean|rho-1|':>13} "
          f"{'std(rho)':>11} {'frac rho>1':>11} {'signed/unsigned':>16}")
    for amp in args.amplitudes:
        g.manual_seed(0)
        noise = (torch.rand(x0.shape, generator=g, device=x0.device,
                            dtype=x0.dtype) - 0.5) * 2.0 * amp * dx
        state.positions = x0 + noise
        rho = computeDensities(state, ctx.config, ctx.schemeConfig, None)
        dev = rho - rho0
        print(f"{amp:10.3f} {dev.mean().item():13.4e} {dev.abs().mean().item():13.4e} "
              f"{rho.std().item():11.4e} {(rho > rho0).float().mean().item():11.4f} "
              f"{(dev.mean() / dev.abs().mean()).item():16.4f}")
    raise _Done


mod.buildSystem = _wrapped
mod.kolmogorovIncompressibleCase.buildSystem = _wrapped

try:
    caseMain(mod.kolmogorovIncompressibleCase, argv=[
        '--nx', str(args.nx), '--nSteps', '1', '--tLimit', '1000.0',
        '--quiet', '--no-store', '--no-plot',
    ])
except _Done:
    pass
