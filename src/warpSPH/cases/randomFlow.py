"""Decaying random flow (2D), weakly compressible.

The script forms of this case were
`examples/weaklyCompressible/06-periodic-random-flow.ipynb` and
`07-bounded-random-flow.ipynb`, plus their incompressible twin
`examples/incompressible/periodic-random-flow.ipynb`. All three seed the same
divergence-free noise field; they differ only in whether the box has walls
(`--bounded`) and which scheme integrates it (`--scheme`).

The noise is divergence-free by construction, so the initial state is a valid
incompressible field and the run measures how the scheme decays it rather than
how it recovers from a bad start.
"""

from __future__ import annotations

from typing import Dict

from ..modules.noise.sampleDivergenceFree import sampleDivergenceFreeNoise
from ..runner import Case, RunContext, caseMain, registerCase
from .plotting import particlePlot
from .weaklyCompressible import (VELOCITY_DENSITY_FIELDS, WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 WEAKLY_COMPRESSIBLE_PARAMS, boundaryRegion,
                                 buildRegionSystem, configureWeaklyCompressible,
                                 domainBoundarySdf, domainFluidSdf, fluidRegion,
                                 paramExtraData, setupTimestep, shapeSdf,
                                 weaklyCompressibleDiagnostics)

__all__ = ['randomFlowCase', 'noiseVelocities']


def configureScheme(ctx: RunContext) -> None:
    configureWeaklyCompressible(ctx)
    # The surface-detection bandwidth is measured in particle spacings.
    ctx.schemeConfig.bandwith = ctx.spec.L / ctx.param('bandWidth') / ctx.config.dx


def buildSystem(ctx: RunContext):
    regions = [fluidRegion(ctx, domainFluidSdf(ctx))]
    if ctx.param('bounded'):
        regions.append(boundaryRegion(ctx, domainBoundarySdf(ctx)))
    if ctx.param('obstacle'):
        regions.append(boundaryRegion(ctx, shapeSdf('circle', ctx.param('obstacleRadius'))))
    return buildRegionSystem(ctx, regions)


def noiseVelocities(ctx: RunContext, system):
    """The divergence-free Perlin field the notebooks seeded the run with."""
    return sampleDivergenceFreeNoise(
        system.state, ctx.config.domain, ctx.config, ctx.schemeConfig,
        ctx.spec.nx * 2,
        octaves=ctx.param('octaves'), lacunarity=ctx.param('lacunarity'),
        persistence=ctx.param('persistence'), baseFrequency=ctx.param('baseFrequency'),
        tileable=ctx.param('tileable'), kind=ctx.param('kind'), seed=ctx.param('seed'))


def initialConditions(ctx: RunContext, system) -> None:
    system.state.velocities[:] = noiseVelocities(ctx, system)
    setupTimestep(ctx, system)


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return weaklyCompressibleDiagnostics(ctx, state)


randomFlowCase = registerCase(Case(
    name='randomFlow',
    scheme='deltaSPH',
    description='Decaying divergence-free random flow (2D), periodic or bounded.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='06-randomFlow',
        nx=128,
        L=2.0,
        tLimit=10.0,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        targetDt=0.00025,
        # `--bounded` is the 07 notebook; `--obstacle` adds the circular
        # cylinder both notebooks carried but only 06 switched on.
        bounded=False,
        obstacle=False,
        obstacleRadius=0.25,
        bandWidth=16.0,
        octaves=3,
        lacunarity=2,
        persistence=0.5,
        baseFrequency=2,
        tileable=True,
        kind='perlin',
        seed=45906734,
        markerSize=8,
    ),
))


if __name__ == '__main__':
    caseMain(randomFlowCase)
