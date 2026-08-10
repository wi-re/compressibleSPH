"""Rayleigh-Taylor instability (2D), compressible.

The script form of this case was
`examples/compressible/13-Rayleigh_Taylor.ipynb`. Heavy fluid over light in a
tall box under constant gravity; the sampler installs both the Dirichlet
boundary bands and the gravity forcing, so `buildSystem` is all this case needs.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..caseUtils import sampleRayleighTaylor
from ..runner import Case, RunContext, caseMain, registerCase
from .compressible import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, configureCompressible,
                           paramExtraData)
from .plotting import Field, particlePlot

__all__ = ['rayleighTaylorCase']


def configureScheme(ctx: RunContext) -> None:
    # A tall box, `aspect` times narrower than it is high, with `band` particle
    # layers of Dirichlet buffer added above and below.
    dx = ctx.spec.L / ctx.spec.nx
    band = ctx.param('band') * dx
    aspect = ctx.param('aspect')
    domain = ctx.config.domain
    domain.min = torch.tensor([-ctx.spec.L / 2 / aspect, -band],
                              device=ctx.device, dtype=ctx.dtype)
    domain.max = torch.tensor([ctx.spec.L / 2 / aspect, ctx.spec.L + band],
                              device=ctx.device, dtype=ctx.dtype)
    configureCompressible(ctx)


def buildSystem(ctx: RunContext):
    return sampleRayleighTaylor(
        ctx.param('rho_low'), ctx.param('rho_high'), ctx.param('delta'),
        ctx.param('g'), ctx.spec.L, ctx.spec.L / ctx.spec.nx, ctx.param('aspect'),
        ctx.spec.nx, ctx.config, ctx.schemeConfig,
        ctx.SimulationState, ctx.SimulationSystem)


setupPlot, updatePlot = particlePlot([
    Field('velocities', 'Velocity', colorMap='viridis', mapping='L2',
          vMin=0.0, vMax=1 / np.sqrt(2)),
    Field('densities', 'density', colorMap='RdBu', colorMapKind='diverging',
          flip=True, gridResolution=1024, vMin=0.95, vMax=2.05),
], figsize=(12, 12))


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return dict(compressibleDiagnostics(ctx, state),
                maxVelocity=torch.linalg.norm(state.state.velocities, dim=-1)
                .max().detach().cpu().item())


rayleighTaylorCase = registerCase(Case(
    name='rayleighTaylor',
    scheme='CRKSPH',
    description='Rayleigh-Taylor instability (2D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='13-rayleighTaylor',
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
        rho_low=1.0,
        rho_high=2.0,
        delta=0.0025,
        g=0.5,
        aspect=2.0,
        band=20,
    ),
))


if __name__ == '__main__':
    caseMain(rayleighTaylorCase)
