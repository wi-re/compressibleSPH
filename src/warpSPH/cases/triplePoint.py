"""Triple-point shock interaction (2D), compressible.

The script forms of this case were `examples/compressible/14-Triple_point.ipynb`
and `15-Triple_point_equalMass.ipynb`. Both run the same three-region initial
state in the same 14x6 box; they differ only in how the regions are sampled --
one particle spacing everywhere (`--no-equalMass`), or one particle *mass*
everywhere, which means the light region is sampled sqrt(8) times coarser
(`--equalMass`).

Notebook 15 also carried an inline copy of `sampleTriplePointEqualMass`,
re-deriving the region masks by hand after calling the helper. The packaged
helper is used here instead; the two agree apart from the sign of `splitY`,
which is symmetric anyway.
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from ..caseUtils import sampleTriplePointEqualMass, sampleTriplePointEqualResolution
from ..runner import Case, RunContext, caseMain, registerCase
from .compressible import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, compressibleTimestep,
                           configureCompressible, paramExtraData)
from .plotting import Field, particlePlot

__all__ = ['triplePointCase']


def configureScheme(ctx: RunContext) -> None:
    domain = ctx.config.domain
    domain.min = torch.tensor([0.0, -3.0], device=ctx.device, dtype=ctx.dtype)
    domain.max = torch.tensor([14.0, 3.0], device=ctx.device, dtype=ctx.dtype)
    configureCompressible(ctx)


def buildSystem(ctx: RunContext):
    states = dict(
        rho_I=ctx.param('rho_I'), p_I=ctx.param('p_I'),
        rho_II=ctx.param('rho_II'), p_II=ctx.param('p_II'),
        rho_III=ctx.param('rho_III'), p_III=ctx.param('p_III'),
    )
    common = dict(
        splitX=ctx.param('splitX'), splitY=ctx.param('splitY'),
        config=ctx.config, schemeConfig=ctx.schemeConfig,
        extraData=dict(ctx.spec.params),
        SimulationState=ctx.SimulationState, SimulationSystem=ctx.SimulationSystem,
        **states,
    )
    if ctx.param('equalMass'):
        # sqrt(8) is the density ratio between regions I/III and region II, so
        # sampling region II that much coarser equalises the particle masses.
        ratio = math.sqrt(8)
        nxs = [ctx.spec.nx * ratio, ctx.spec.nx, ctx.spec.nx * ratio, ctx.spec.nx * ratio]
        return sampleTriplePointEqualMass(nxs=nxs, **common)
    return sampleTriplePointEqualResolution(nx=ctx.spec.nx, **common)


setupPlot, updatePlot = particlePlot([
    Field('densities', 'Density field (log scale)', colorMap='RdBu',
          colorMapKind='diverging', flip=True, scaling='Logarithmic',
          gridResolution=1024, vMin=0.2, vMax=7.0),
], figsize=(12, 6))


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return dict(compressibleDiagnostics(ctx, state),
                maxVelocity=torch.linalg.norm(state.state.velocities, dim=-1)
                .max().detach().cpu().item())


triplePointCase = registerCase(Case(
    name='triplePoint',
    scheme='CRKSPH',
    description='Triple-point shock interaction (2D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    diagnostics=diagnostics,
    # Only meaningful for the equal-mass sampling, where the coarse region's
    # sound speed changes a lot; equal-resolution runs pin dt below instead.
    timestep=compressibleTimestep,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='14-triplePoint',
        dim=2,
        nx=128,
        L=1.0,
        tLimit=10.0,
        plotInterval=1,
        storeInterval=500,
    ),
    params=dict(
        COMPRESSIBLE_PARAMS,
        gamma=1.4,
        equalMass=True,
        splitX=1.0,
        splitY=1.5,
        rho_I=1.0, p_I=1.0,
        rho_II=0.125, p_II=0.1,
        rho_III=1.0, p_III=0.1,
        aspect=2.0,
        band=20,
    ),
))


if __name__ == '__main__':
    caseMain(triplePointCase)
