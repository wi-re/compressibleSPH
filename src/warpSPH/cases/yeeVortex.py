"""Yee isentropic vortex (2D), compressible.

The script form of this case was `examples/compressible/10-Yee_Vortex.ipynb`.
The vortex is sampled on concentric shells rather than a lattice, and the outer
`bufferRings` shells are held at their initial state by a Dirichlet boundary
condition -- the sampler returns that BC, which is why it is installed here
rather than being part of the scheme config.
"""

from __future__ import annotations

from typing import Dict

from ..caseUtils import sampleYeeVortex
from ..modules import enforceDirichlet
from ..runner import Case, RunContext, caseMain, registerCase
from .compressible import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, configureCompressible,
                           paramExtraData)
from .plotting import Field, particlePlot

__all__ = ['yeeVortexCase']


def buildSystem(ctx: RunContext):
    # `sampleYeeVortex` reads its physical parameters out of a dict, which is
    # exactly what `spec.params` is.
    system, indices, boundary = sampleYeeVortex(
        ctx.param('nr'), ctx.spec.nx, ctx.param('bufferRings'),
        ctx.config, ctx.schemeConfig, dict(ctx.spec.params),
        ctx.SimulationState, ctx.SimulationSystem)
    ctx.scratch['shellIndices'] = indices

    ctx.schemeConfig.boundaryConditions.clear()
    ctx.schemeConfig.boundaryConditions.append(boundary)
    enforceDirichlet(system, system.t, ctx.config.dt, ctx.config, ctx.schemeConfig)
    return system


setupPlot, updatePlot = particlePlot([
    Field('velocities', 'velocities', colorMap='viridis', mapping='L2Norm'),
    Field('densities', 'densities', colorMap='cividis', flip=True),
])


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return compressibleDiagnostics(ctx, state)


yeeVortexCase = registerCase(Case(
    name='yee',
    scheme='CRKSPH',
    description='Yee isentropic vortex (2D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureCompressible,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='10-yeeVortex',
        dim=2,
        nx=200,
        L=10.0,
        tLimit=8.0,
        plotInterval=1,
        storeInterval=500,
    ),
    params=dict(
        COMPRESSIBLE_PARAMS,
        gamma=1.4,
        nr=32,
        bufferRings=10,
        xc=0.0,
        yc=0.0,
        beta=5.0,
        P_infty=1.0,
        rho_infty=1.0,
        markerSize=4,
    ),
))


if __name__ == '__main__':
    caseMain(yeeVortexCase)
