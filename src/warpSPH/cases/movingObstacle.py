"""Flow past a spinning obstacle (2D), weakly compressible.

The script form of this case was
`examples/weaklyCompressible/10-moving-obstacle.ipynb`. A hexagonal rigid body
spins in a periodic box while the domain-mean velocity is driven towards
`U_target`.

The forcing acts on the *mean* velocity only, deliberately: forcing every
particle towards the target would damp the wake fluctuations that are the whole
point of the case.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..configurations import BoundaryCondition, BoundaryConditionType
from ..configurations.region import BCType
from ..runner import Case, RunContext, caseMain, registerCase
from .plotting import particlePlot
from .weaklyCompressible import (VELOCITY_DENSITY_FIELDS, WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 WEAKLY_COMPRESSIBLE_PARAMS, boundaryRegion,
                                 buildRegionSystem, configureWeaklyCompressible,
                                 domainFluidSdf, fluidRegion, paramExtraData,
                                 setupTimestep, shapeSdf,
                                 weaklyCompressibleDiagnostics)

__all__ = ['movingObstacleCase']


def buildSystem(ctx: RunContext):
    fluidSdf = domainFluidSdf(ctx)
    ctx.scratch['fluidSdf'] = fluidSdf
    return buildRegionSystem(ctx, [
        fluidRegion(ctx, fluidSdf),
        boundaryRegion(ctx, shapeSdf(ctx.param('obstacleShape'), ctx.param('obstacleSize')),
                       kind=BCType.constant),
    ])


def initialConditions(ctx: RunContext, system) -> None:
    setupTimestep(ctx, system)

    ctx.config.rigidBodies[0].angularVelocity = ctx.param('obstacleOmega')
    ctx.schemeConfig.rigidBodies = ctx.config.rigidBodies

    target = ctx.param('U_target')
    tau = ctx.param('forcingTau')

    def meanFlowForcing(state, config, schemeConfig, positions, d, n, t, dt):
        force = torch.zeros_like(state.positions)
        fluid = state.kinds == 0
        if torch.count_nonzero(fluid) == 0:
            return force
        mean = state.velocities[fluid].mean(dim=0)
        force[fluid, 0] = state.masses[fluid] * (target - mean[0]) / tau
        force[fluid, 1] = state.masses[fluid] * (-mean[1]) / tau
        return force

    ctx.schemeConfig.boundaryConditions = [BoundaryCondition(
        type=BoundaryConditionType.dynamic,
        sdf=ctx.scratch['fluidSdf'],
        forcingFunctions=[meanFlowForcing],
    )]


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return weaklyCompressibleDiagnostics(ctx, state)


movingObstacleCase = registerCase(Case(
    name='movingObstacle',
    scheme='deltaSPH',
    description='Flow past a spinning rigid obstacle (2D), weakly compressible deltaSPH.',
    buildSystem=buildSystem,
    configureScheme=configureWeaklyCompressible,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='10-movingObstacle',
        nx=128,
        L=2.0,
        tLimit=10.0,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        obstacleShape='hexagon',
        obstacleSize=0.25,
        obstacleOmega=1.0,
        U_target=1.0,
        forcingTau=0.5,
        markerSize=8,
    ),
))


if __name__ == '__main__':
    caseMain(movingObstacleCase)
