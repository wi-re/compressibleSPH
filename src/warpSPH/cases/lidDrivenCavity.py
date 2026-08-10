"""Lid-driven cavity (2D), weakly compressible.

The script form of this case was `examples/weaklyCompressible/09-LDC.ipynb`.
A no-slip box whose top wall is dragged sideways at unit speed, imposed as a
dynamic Dirichlet condition on the velocity rather than as a moving boundary
region.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..configurations import BoundaryCondition, BoundaryConditionType
from ..configurations.region import BCType
from ..modules import enforceDirichlet
from ..runner import Case, RunContext, caseMain, registerCase
from .plotting import particlePlot
from .weaklyCompressible import (VELOCITY_DENSITY_FIELDS, WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 WEAKLY_COMPRESSIBLE_PARAMS, boundaryRegion,
                                 buildRegionSystem, configureWeaklyCompressible,
                                 domainBoundarySdf, domainFluidSdf, fluidRegion,
                                 paramExtraData, setupTimestep,
                                 weaklyCompressibleDiagnostics)

__all__ = ['lidDrivenCavityCase']


def buildSystem(ctx: RunContext):
    fluidSdf = domainFluidSdf(ctx)
    ctx.scratch['fluidSdf'] = fluidSdf
    return buildRegionSystem(ctx, [
        fluidRegion(ctx, fluidSdf),
        boundaryRegion(ctx, domainBoundarySdf(ctx), kind=BCType.noSlip),
    ])


def initialConditions(ctx: RunContext, system) -> None:
    setupTimestep(ctx, system)

    lidHeight = ctx.param('lidHeight')
    lidVelocity = ctx.param('lidVelocity')

    def lidDirichlet(state, config, schemeConfig, positions, d, n, t, dt):
        velocities = state.velocities.clone()
        velocities[:, 0] = torch.where(positions[:, 1] > lidHeight, lidVelocity,
                                       velocities[:, 0])
        return velocities

    ctx.schemeConfig.boundaryConditions = [BoundaryCondition(
        type=BoundaryConditionType.dynamic,
        sdf=ctx.scratch['fluidSdf'],
        dirichletFunctions={'velocities': lidDirichlet},
    )]
    enforceDirichlet(system, system.t, ctx.config.dt, ctx.config, ctx.schemeConfig)


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return weaklyCompressibleDiagnostics(ctx, state)


lidDrivenCavityCase = registerCase(Case(
    name='ldc',
    scheme='deltaSPH',
    description='Lid-driven cavity (2D), weakly compressible deltaSPH.',
    buildSystem=buildSystem,
    configureScheme=configureWeaklyCompressible,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='09-lidDrivenCavity',
        nx=128,
        L=2.0,
        # Long enough for the primary vortex to reach steady state.
        tLimit=30.0,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        band=5,
        lidHeight=1.0,
        lidVelocity=1.0,
        markerSize=8,
    ),
))


if __name__ == '__main__':
    caseMain(lidDrivenCavityCase)
