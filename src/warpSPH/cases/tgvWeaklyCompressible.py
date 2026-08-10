"""Taylor-Green vortex (2D), weakly compressible.

The script form of this case was
`examples/weaklyCompressible/05-taylor-green-vortex.ipynb` -- the same vortex as
`warpSPH.cases.tgv`, but integrated by deltaSPH with an explicit physical
viscosity instead of by the divergence-free incompressible solver.

The analytic answer is `KE(t) = KE(0) exp(-4 nu k^2 t)`, which
:func:`effectiveViscosity` fits back out of the run. The measured value lands
consistently *below* the prescribed one: the diffusion operator carries a
Monaghan switch that turns viscosity off for particle pairs that are separating,
so roughly half of the pairs at any instant contribute no dissipation. Disabling
the switch recovers the analytic decay rate but costs stability elsewhere, which
is why it stays on.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..modules import shuffleParticles
from ..runner import Case, RunContext, caseMain, registerCase
from ..sample.weaklyCompressible import setupBasicWeaklyCompressibleInitialState
from .plotting import particlePlot
from .weaklyCompressible import (VELOCITY_DENSITY_FIELDS, WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 WEAKLY_COMPRESSIBLE_PARAMS,
                                 configureWeaklyCompressible, paramExtraData,
                                 setupTimestep, weaklyCompressibleDiagnostics)

__all__ = ['tgvWeaklyCompressibleCase', 'effectiveViscosity', 'analyticDecayRate']


def wavenumber(ctx: RunContext) -> float:
    """The TGV wavenumber actually stamped onto the velocity field."""
    return ctx.param('k') / 2.0


def analyticDecayRate(ctx: RunContext) -> float:
    """`4 nu k^2`, the exponential rate of the kinetic-energy decay."""
    return 4.0 * ctx.param('nu') * wavenumber(ctx) ** 2


def effectiveViscosity(result) -> float:
    """Fit `nu_eff` from a completed run's kinetic-energy history."""
    ts = result.series('t')
    energies = result.series('kineticEnergy')
    mask = (ts > 0) & (energies > 0)
    slope = np.polyfit(ts[mask], np.log(energies[mask] / energies[0]), 1)[0]
    k = result.ctx.param('k') / 2.0
    return -slope / (4 * k ** 2)


def configureScheme(ctx: RunContext) -> None:
    configureWeaklyCompressible(ctx)
    # The TGV box is [0, L]^2, not the symmetric box the shared block builds.
    domain = ctx.config.domain
    domain.min = torch.zeros(ctx.spec.dim, device=ctx.device, dtype=ctx.dtype)
    domain.max = torch.ones(ctx.spec.dim, device=ctx.device, dtype=ctx.dtype) * ctx.spec.L


def buildSystem(ctx: RunContext):
    system = setupBasicWeaklyCompressibleInitialState(
        ctx.spec.nx, ctx.config, ctx.schemeConfig, ctx.SimulationState, ctx.SimulationSystem)
    # A perfectly regular lattice is an unstable SPH equilibrium; the shuffle is
    # what keeps the early trajectory free of lattice noise.
    if ctx.param('shuffleIters'):
        system.state.positions = shuffleParticles(
            system.state, ctx.config, ctx.schemeConfig, ctx.param('shuffleIters'),
            jitterAmount=ctx.param('jitter'))
    return system


def initialConditions(ctx: RunContext, system) -> None:
    k = wavenumber(ctx)
    uMag = ctx.param('uMag')
    # An even wavenumber puts the vortex centres on the domain boundary; the
    # quarter-period shift moves them back into the interior.
    phase = np.pi / 2 if ctx.param('k') % 2 == 0 else 0.0

    positions = system.state.positions
    system.state.velocities[:, 0] = (
        uMag * torch.cos(k * positions[:, 0] + phase) * torch.sin(k * positions[:, 1] + phase))
    system.state.velocities[:, 1] = (
        -uMag * torch.sin(k * positions[:, 0] + phase) * torch.cos(k * positions[:, 1] + phase))

    setupTimestep(ctx, system)


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return weaklyCompressibleDiagnostics(ctx, state)


tgvWeaklyCompressibleCase = registerCase(Case(
    name='tgv-wc',
    scheme='deltaSPH',
    description='Taylor-Green vortex (2D), weakly compressible deltaSPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='05-taylorGreenVortex',
        nx=256,
        L=2 * np.pi,
        tLimit=2.0,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        targetDt=0.001,
        inviscid=False,
        nu=0.01,
        k=2,
        uMag=1.0,
        shuffleIters=128,
        jitter=1.0,
    ),
))


if __name__ == '__main__':
    caseMain(tgvWeaklyCompressibleCase)
