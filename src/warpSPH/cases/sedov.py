"""Sedov-Taylor blast wave, compressible.

The script forms of this case were
`examples/compressible/06-Sedov_Taylor_Blastwave_1D.ipynb` and
`07-Sedov_Taylor_Blastwave_2D.ipynb`. They are the *same* case at two
dimensionalities -- identical sampler, identical scheme, identical stopping
rule -- so this is one case run as ``--dim 1`` or ``--dim 2``, and only the
plot branches on it.

The run ends when the shock reaches `goalRadius`, which is a time derived from
the analytic self-similar solution rather than a number chosen by hand.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..caseUtils import SedovSolution, beta, buildSedov, radius
from ..runner import Case, RunContext, caseMain, registerCase
from .compressible import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, configureCompressible,
                           paramExtraData)
from .plotting import Field, ProfileAxis, particlePlot, profilePlot

__all__ = ['sedovCase', 'sedovSolution', 'goalTime']


def sedovSolution(ctx: RunContext) -> SedovSolution:
    return SedovSolution(
        nDim=ctx.spec.dim,
        gamma=ctx.param('gamma'),
        rho0=ctx.param('rho0'),
        E0=ctx.param('E0'),
        h0=2 / ctx.spec.nx,
    )


def goalTime(ctx: RunContext, solution: SedovSolution) -> float:
    """The time at which the shock front reaches ``goalRadius``."""
    nu1 = 1.0 / (solution.nu + 2.0)
    return (ctx.param('goalRadius')
            * (solution.alpha * ctx.param('rho0') / ctx.param('E0')) ** nu1) ** (1.0 / (2.0 * nu1))


def buildSystem(ctx: RunContext):
    solution = sedovSolution(ctx)
    ctx.scratch['solution'] = solution
    ctx.spec = ctx.spec.merged(tLimit=goalTime(ctx, solution))

    return buildSedov(
        ctx.SimulationSystem, ctx.SimulationState,
        config=ctx.config,
        nx=ctx.spec.nx, dim=ctx.spec.dim, domainExtent=ctx.spec.L,
        periodicDomain=ctx.spec.periodic,
        rho0=ctx.param('rho0'), E0=ctx.param('E0'),
        initialization=ctx.param('initialization'),
        gamma=ctx.param('gamma'), kernel=ctx.config.kernel,
        targetNeighbors=ctx.config.targetNeighbors,
        dtype=ctx.config.dtype, device=ctx.config.device)


# -- plotting ----------------------------------------------------------------
# 1D shows radial profiles against the analytic shock state; 2D shows the two
# fields the blast is actually read from. `setupPlot` picks between them at run
# time because the dimension is not known until the spec is resolved.

def _shock(ctx: RunContext, state):
    """`(shockSpeed, r2, v2, rho2, P2)` and the front radius at the current t."""
    solution = ctx.scratch['solution']
    t = max(float(state.t), 1e-12)
    return solution.shockState(t), radius(beta(ctx.spec.dim), ctx.param('E0'), t,
                                          ctx.param('rho0'), 1)


def _fronts(ctx: RunContext, state):
    (_, r2, _, _, _), rt = _shock(ctx, state)
    return [r2, -r2, rt, -rt]


def _shockDensity(ctx: RunContext, state):
    (_, _, _, rho2, _), _ = _shock(ctx, state)
    return [ctx.param('rho0'), rho2]


def _shockVelocity(ctx: RunContext, state):
    (vs, _, v2, _, _), _ = _shock(ctx, state)
    return [vs, -vs, v2, -v2]


_profileSetup, _profileUpdate = profilePlot(
    [
        ProfileAxis('internalEnergies', 'Internal energy', yscale='log', vlines=_fronts),
        ProfileAxis('densities', 'Density', yscale='log', vlines=_fronts,
                    hlines=_shockDensity),
        ProfileAxis('velocities', 'Velocity', component=0, vlines=_fronts,
                    hlines=_shockVelocity),
        ProfileAxis('supports', 'Support', vlines=_fronts),
    ],
    shape=(2, 2), figsize=(9, 6), xlim=(0, 1),
)

_fieldSetup, _fieldUpdate = particlePlot([
    Field('internalEnergies', 'internal energy', colorMap='viridis',
          scaling='Logarithmic', vMin=1e-10, gridResolution=1024),
    Field('densities', 'density', colorMap='cividis'),
])


def setupPlot(ctx: RunContext, state):
    ctx.scratch['plotters'] = ((_profileSetup, _profileUpdate) if ctx.spec.dim == 1
                               else (_fieldSetup, _fieldUpdate))
    return ctx.scratch['plotters'][0](ctx, state)


def updatePlot(ctx: RunContext, state, handle, step: int) -> None:
    ctx.scratch['plotters'][1](ctx, state, handle, step)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return compressibleDiagnostics(ctx, state)


sedovCase = registerCase(Case(
    name='sedov',
    scheme='CRKSPH',
    description='Sedov-Taylor blast wave (1D or 2D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureCompressible,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='06-sedovTaylorBlastwave',
        # 1D was the notebook's nx=800; the 2D notebook used nx=200, which is
        # `--dim 2 --nx 200`.
        dim=1,
        nx=800,
        L=2.0,
        # Replaced in buildSystem by the analytic time to reach goalRadius.
        tLimit=1.0,
        plotInterval=25,
        storeInterval=500,
    ),
    params=dict(
        COMPRESSIBLE_PARAMS,
        E0=1.0,
        goalRadius=0.8,
        # Both notebooks asked for 'hat' and both therefore crashed:
        # `buildSedov` raises NotImplementedError for it, and the dead code
        # below that raise calls `warpKernelToDiffSPHKernel`/`diffSPHKernel`,
        # names left over from the pre-warp stack that no longer exist
        # anywhere. 'singular' deposits E0 on the single central particle and
        # is what actually runs; 'quadrant' spreads it over the 2^dim
        # innermost particles.
        initialization='singular',
    ),
))


if __name__ == '__main__':
    caseMain(sedovCase)
