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

That makes this case the family's viscosity calibration: it is the one setup
whose dissipation has a closed-form answer, so it is where "what did I ask for"
(`nu`, or `alpha` when `--inviscid`; :func:`viscosityScales` converts between
them) can be held against "what did the scheme actually apply"
(:func:`effectiveViscosity`). Sweeping `nu` over decades and fitting each run is
the notebook's last section.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..modules import alphaToNu, nuToAlpha, shuffleParticles
from ..runner import Case, RunContext, caseMain, registerCase
from ..sample.weaklyCompressible import setupBasicWeaklyCompressibleInitialState
from .plotting import particlePlot
from .weaklyCompressible import (VELOCITY_DENSITY_FIELDS, WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 WEAKLY_COMPRESSIBLE_PARAMS,
                                 configureWeaklyCompressible, paramExtraData,
                                 setupTimestep, weaklyCompressibleDiagnostics)

__all__ = ['tgvWeaklyCompressibleCase', 'effectiveViscosity', 'analyticDecayRate',
           'analyticKineticEnergy', 'viscosityScales', 'wavenumber',
           'MIN_STABLE_ALPHA']

#: Empirically, an artificial viscosity below this stops being reliably stable.
#: It is what makes "the largest Reynolds number this discretisation can carry"
#: a computable number rather than a matter of taste -- see
#: :func:`viscosityScales`.
MIN_STABLE_ALPHA = 0.01


def wavenumber(ctx: RunContext) -> float:
    """The TGV wavenumber actually stamped onto the velocity field."""
    return ctx.param('k') / 2.0


def analyticDecayRate(ctx: RunContext) -> float:
    """`4 nu k^2`, the exponential rate of the kinetic-energy decay."""
    return 4.0 * ctx.param('nu') * wavenumber(ctx) ** 2


def analyticKineticEnergy(ctx: RunContext) -> float:
    """`KE(0)` of the continuum vortex, `rho0 uMag^2 L^2 / 4`.

    The mean of `u^2 + v^2` over a whole number of periods of the TGV field is
    `uMag^2 / 2`, so the continuum answer needs no integration -- and comparing
    it to what the sampled particles actually carry is the cheapest check that
    the lattice, the masses and the shuffle all came out right.
    """
    return 0.25 * ctx.param('uMag') ** 2 * ctx.param('rho0') * ctx.spec.L ** 2


def viscosityScales(ctx: RunContext, state) -> Dict[str, float]:
    """The viscosity of the run as configured, in every form it has one.

    `nu` and `alpha` are the same dissipation written two ways -- physical
    kinematic viscosity and the deltaSPH artificial-viscosity coefficient --
    related by `nu = alpha c0 h / (2(n+2))` (Sun et al. 2016 against Marrone et
    al. 2012). Whichever one the case was given, the other follows once the
    sound speed and the mean support radius exist, which is why this takes a
    state rather than only the spec -- the built system before the run, or a
    `RunResult.state` after it; anything carrying `.state.supports`.

    Also returns the Reynolds number that implies, and the largest one this
    discretisation can carry: `alpha` cannot usefully go below
    :data:`MIN_STABLE_ALPHA`, and that floor is a viscosity floor, hence a
    Reynolds ceiling.
    """
    dim = ctx.spec.dim
    c0 = float(ctx.schemeConfig.fluid.fixedSoundSpeed)
    h = float(state.state.supports.mean().detach().cpu())
    # The velocity scale is uMag and the length scale is half the box: one
    # vortex, not the periodic tile that holds four of them.
    scale = ctx.param('uMag') * ctx.spec.L / 2

    if ctx.param('inviscid'):
        alpha = ctx.param('alpha')
        nu = alphaToNu(alpha, c0, h, dim)
    else:
        nu = ctx.param('nu')
        alpha = nuToAlpha(nu, c0, h, dim)
    nuLimit = alphaToNu(MIN_STABLE_ALPHA, c0, h, dim)

    return {
        'nu': nu, 'alpha': alpha, 'c0': c0, 'h': h,
        'Re': scale / nu if nu > 0 else float('inf'),
        'nuLimit': nuLimit, 'ReLimit': scale / nuLimit,
    }


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
