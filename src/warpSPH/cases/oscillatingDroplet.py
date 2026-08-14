"""Oscillating droplet under a central potential (2D), weakly compressible.

The script form of this case was
`examples/weaklyCompressible/04-oscillating-droplet.ipynb`. A circular droplet
is given a straining velocity field and held together by a radial potential;
the exact solution oscillates between two ellipses with period T = 4.827 A, and
that period is what this run is measured against.
"""

from __future__ import annotations

from typing import Dict

from ..runner import Case, RunContext, caseMain, registerCase
from .plotting import particlePlot
from .weaklyCompressible import (VELOCITY_DENSITY_FIELDS, WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 WEAKLY_COMPRESSIBLE_PARAMS, buildRegionSystem,
                                 configureWeaklyCompressible, fluidRegion,
                                 paramExtraData, setupTimestep, shapeSdf,
                                 weaklyCompressibleDiagnostics)

__all__ = ['oscillatingDropletCase', 'DROPLET_STRETCH', 'DROPLET_PERIOD',
           'analyticEnvelope']

#: Long semi-axis of the droplet at maximum elongation, in units of `R`.
#: The short one is `R / DROPLET_STRETCH`: the flow is incompressible, so the
#: ellipse has the same area as the circle it started as.
DROPLET_STRETCH = 1.931843

#: Oscillation period, in units of the strain time `1 / A`.
DROPLET_PERIOD = 4.827


def analyticEnvelope(R: float = 1.0, stretch: float = DROPLET_STRETCH):
    """`(long, short)` semi-axes of the two extreme ellipses.

    The droplet passes through the circle it started as and through two
    ellipses that are each other's 90-degree rotation -- long axis horizontal
    at one extreme, vertical at the other. `04-oscillating-droplet.ipynb` draws
    all three over the particles as the check that the amplitude, not just the
    period, came out right.
    """
    return stretch * R, R / stretch


def configureScheme(ctx: RunContext) -> None:
    # The restoring force is a potential field centred on the droplet, not a
    # directional gravity, so it is configured before the shared block runs.
    ctx.spec.params.setdefault('gravity', True)
    ctx.spec.params.setdefault('gravityType', 'PotentialField')
    ctx.spec.params.setdefault('gravityMagnitude', ctx.param('B'))
    ctx.spec.params.setdefault('gravityDirection', [0.0, 0.0])
    configureWeaklyCompressible(ctx)


def buildSystem(ctx: RunContext):
    return buildRegionSystem(
        ctx, [fluidRegion(ctx, shapeSdf('circle', ctx.param('R')))])


def initialConditions(ctx: RunContext, system) -> None:
    strain = ctx.param('A')
    positions = system.state.positions
    system.state.velocities[:, 0] = strain * positions[:, 0]
    system.state.velocities[:, 1] = -strain * positions[:, 1]
    setupTimestep(ctx, system)


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return weaklyCompressibleDiagnostics(ctx, state)


oscillatingDropletCase = registerCase(Case(
    name='droplet',
    scheme='deltaSPH',
    description='Oscillating droplet in a central potential (2D), weakly compressible deltaSPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='04-oscillatingDroplet',
        nx=192,
        L=6.0,
        # One full oscillation period, DROPLET_PERIOD / A with A = 1.
        tLimit=DROPLET_PERIOD,
        plotInterval=10,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        freeSurface=True,
        targetDt=0.00025,
        R=1.0,
        A=1.0,
        B=1.0,
        markerSize=8,
    ),
))


if __name__ == '__main__':
    caseMain(oscillatingDropletCase)
