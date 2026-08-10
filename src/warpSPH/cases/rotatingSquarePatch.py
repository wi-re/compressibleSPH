"""Rotating square patch (2D), weakly compressible.

The script forms of this case were
`examples/weaklyCompressible/03-rotating-square-patch.ipynb` and
`examples/incompressible/03-rotating-square-patch.ipynb` -- the same geometry
under two schemes, which is `--scheme deltaSPH` or `--scheme divergenceFree`
here.

A square of fluid in rigid rotation is not an equilibrium: the free surface has
to deform, and the arms it grows are the thing being compared between schemes.
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

__all__ = ['rotatingSquarePatchCase']


def buildSystem(ctx: RunContext):
    half = ctx.param('halfExtent')
    return buildRegionSystem(ctx, [fluidRegion(ctx, shapeSdf('box', [half, half]))])


def initialConditions(ctx: RunContext, system) -> None:
    omega = ctx.param('omega')
    positions = system.state.positions
    system.state.velocities[:, 0] = omega * positions[:, 1]
    system.state.velocities[:, 1] = -omega * positions[:, 0]
    setupTimestep(ctx, system)


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return weaklyCompressibleDiagnostics(ctx, state)


rotatingSquarePatchCase = registerCase(Case(
    name='squarePatch',
    scheme='deltaSPH',
    description='Rotating square patch of fluid (2D), weakly compressible or incompressible.',
    buildSystem=buildSystem,
    configureScheme=configureWeaklyCompressible,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='03-rotatingSquarePatch',
        nx=192,
        # 2 * 3 * R: the patch grows arms well past its initial extent, so the
        # box has to be much larger than the patch.
        L=6.0,
        tLimit=1.0,
        plotInterval=10,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        freeSurface=True,
        halfExtent=1.0,
        omega=4.0,
        markerSize=8,
    ),
))


if __name__ == '__main__':
    caseMain(rotatingSquarePatchCase)
