"""Two bodies of fluid colliding head-on (2D), weakly compressible.

The script forms of this case were
`examples/weaklyCompressible/01-impact_spheres.ipynb` and
`02-impact-squares.ipynb`. They are the same experiment with a different
initial shape -- two free-surface bodies given equal and opposite velocities --
so this is one case selected by `--shape circle|box`, and the collision axis
follows the shape the way the notebooks had it (spheres meet along x, squares
along y).
"""

from __future__ import annotations

from typing import Dict

from ..runner import Case, RunContext, caseMain, registerCase
from .plotting import Field, particlePlot
from .weaklyCompressible import (WEAKLY_COMPRESSIBLE_DEFAULTS, WEAKLY_COMPRESSIBLE_PARAMS,
                                 buildRegionSystem, configureWeaklyCompressible,
                                 fluidRegion, paramExtraData, setupTimestep, shapeSdf,
                                 weaklyCompressibleDiagnostics)

__all__ = ['impactCase']


def _geometry(ctx: RunContext):
    """`(axis, sdfA, sdfB)` -- the collision axis and the two bodies."""
    shape = ctx.param('shape')
    offset = ctx.param('offset')
    if shape == 'circle':
        radius = ctx.param('radius')
        return 0, shapeSdf('circle', radius, [-offset, 0.0]), \
            shapeSdf('circle', radius, [offset, 0.0])
    if shape == 'box':
        half = ctx.param('halfExtent')
        # The two boxes are separated by one particle spacing so they start
        # just touching rather than overlapping.
        gap = ctx.config.dx
        return 1, shapeSdf('box', [half, half / 2], [0.0, half / 2 + gap]), \
            shapeSdf('box', [half, half / 2], [0.0, -half / 2 - gap])
    raise ValueError(f"Unknown shape {shape!r}. Known: 'circle', 'box'.")


def buildSystem(ctx: RunContext):
    axis, sdfA, sdfB = _geometry(ctx)
    ctx.scratch['axis'] = axis
    return buildRegionSystem(ctx, [fluidRegion(ctx, sdfA), fluidRegion(ctx, sdfB)])


def initialConditions(ctx: RunContext, system) -> None:
    axis = ctx.scratch['axis']
    speed = ctx.param('impactVelocity')
    positions = system.state.positions
    system.state.velocities[positions[:, axis] < 0, axis] = speed
    system.state.velocities[positions[:, axis] > 0, axis] = -speed
    setupTimestep(ctx, system)


setupPlot, updatePlot = particlePlot([
    Field('velocities', 'velocities', colorMap='viridis', mapping='L2Norm'),
    Field('densities', 'densities', colorMap='flare', flip=True, midPoint=1.0,
          vMin=0.99, vMax=1.01),
])


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return weaklyCompressibleDiagnostics(ctx, state)


impactCase = registerCase(Case(
    name='impact',
    scheme='deltaSPH',
    description='Head-on impact of two fluid bodies (2D), weakly compressible deltaSPH.',
    buildSystem=buildSystem,
    configureScheme=configureWeaklyCompressible,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='01-impact',
        nx=256,
        L=4.0,
        tLimit=10.0,
        # The spheres notebook integrated with symplectic Euler; the squares
        # one used RK2. RK2 is the shared default and works for both.
        integrationScheme='rungeKutta2',
        plotInterval=10,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        freeSurface=True,
        shape='circle',
        radius=0.5,
        offset=0.75,
        halfExtent=1.0,
        impactVelocity=0.5,
    ),
))


if __name__ == '__main__':
    caseMain(impactCase)
