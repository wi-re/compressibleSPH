"""Dam break with optional obstacle (2D), weakly compressible.

The script form of this case was `datagen/weaklyCompressible/generator.py`,
whose ~65 argparse flags are now this case's `params`. The `caseUtils` helpers it
calls still take an argparse-style namespace -- they are shared with the
notebooks -- so :func:`caseArgs` rebuilds one from the spec rather than
rewriting them.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict

import torch

from ..caseUtils import (SimulationProperties, buildDomain, buildPresetObstacles,
                         buildRegions, sampleNoise, setupFreestream, setupKolmogorov)
from ..configurations.moduleConfigurations.gravity import GravityType
from ..initializers import initializeWeaklyCompressibleSimulation
from ..modules import setupWeaklyCompressibleTimestep
from ..runner import Case, RunContext, caseMain, registerCase
from ..runner.display import resolvePlotBackend
from .plotting import openWindow, pumpEvents

__all__ = ['dambreakCase', 'caseArgs', 'simulationProperties']


def freeSurface(ctx: RunContext) -> bool:
    if ctx.param('fillRatio') < 1.0:
        return True
    return not (ctx.param('semiPeriodic') or ctx.param('fullyPeriodic'))


def caseArgs(ctx: RunContext) -> SimpleNamespace:
    """The argparse-shaped namespace `caseUtils` expects, built from the spec."""
    values = dict(ctx.spec.params)
    values.update(
        nx=ctx.spec.nx,
        L=ctx.spec.L,
        band=ctx.param('band'),
        caseName=ctx.spec.caseName,
        timeLimit=ctx.spec.tLimit,
        plot=ctx.spec.plot,
        plotInterval=ctx.spec.plotInterval,
        exportInterval=ctx.spec.exportInterval,
    )
    return SimpleNamespace(**values)


def simulationProperties(ctx: RunContext) -> SimulationProperties:
    return SimulationProperties(
        device=ctx.device,
        dtype=ctx.dtype,
        nx=ctx.spec.nx,
        dim=ctx.spec.dim,
        L=ctx.spec.L,
        W=ctx.param('W'),
        dx=ctx.spec.L / ctx.spec.nx,
        band=ctx.param('band'),
        n_h=ctx.spec.n_h,
        targetDt=ctx.param('targetDt'),
        freeSurface=freeSurface(ctx),
        semiPeriodic=ctx.param('semiPeriodic'),
        fullyPeriodic=ctx.param('fullyPeriodic'),
    )


def configureScheme(ctx: RunContext) -> None:
    args = caseArgs(ctx)
    simSetup = simulationProperties(ctx)
    ctx.scratch['args'] = args
    ctx.scratch['simSetup'] = simSetup

    # buildDomain widens the box by `band` particle layers for the boundary.
    domain, interiorDomain = buildDomain(simSetup)
    ctx.config.domain = domain
    ctx.config.nx = simSetup.nx + 2 * simSetup.band
    ctx.config.dx = simSetup.dx
    ctx.scratch['interiorDomain'] = interiorDomain

    schemeConfig = ctx.schemeConfig
    schemeConfig.surfaceDetectionConfig.active = simSetup.freeSurface
    schemeConfig.gravityConfig.active = not ctx.param('disableGravity')
    schemeConfig.gravityConfig.type = GravityType.Directional
    schemeConfig.gravityConfig.magnitude = ctx.param('gravityMagnitude')
    schemeConfig.gravityConfig.origin = ctx.param('gravityDirection')
    schemeConfig.bandwith = simSetup.L / ctx.param('bandWidth') / ctx.config.dx


def buildSystem(ctx: RunContext):
    args = ctx.scratch['args']
    simSetup = ctx.scratch['simSetup']
    dx = ctx.config.dx

    # Snapping the obstacle to the particle lattice keeps its sampled surface
    # free of the half-cell sliver a non-aligned SDF would produce.
    maxExtent = round(ctx.param('maxExtent') / dx) * dx
    offsetX = round(ctx.param('offsetX') / dx) * dx
    presets = buildPresetObstacles(maxExtent, offsetX, ctx.spec.L,
                                   ctx.param('fillRatio'), ctx.param('aoa'))
    obstacle = presets.get(ctx.param('obstacleType'))
    if obstacle is None:
        raise ValueError(f"Unknown obstacleType {ctx.param('obstacleType')!r}. "
                         f'Known: {sorted(presets)}')
    obstacle['offsetY'] = round(obstacle['offsetY'] / dx) * dx

    ctx.schemeConfig.regions = buildRegions(ctx.config, ctx.schemeConfig, simSetup, args,
                                            ctx.config.domain, ctx.scratch['interiorDomain'],
                                            obstacle)
    ctx.schemeConfig.boundaryConditions = []
    ctx.scratch['obstacle'] = obstacle

    return initializeWeaklyCompressibleSimulation(
        ctx.schemeConfig.regions, ctx.config, ctx.schemeConfig,
        ctx.SimulationSystem, ctx.SimulationState, verbose=ctx.spec.verbose)


def initialConditions(ctx: RunContext, system) -> None:
    args = ctx.scratch['args']
    simSetup = ctx.scratch['simSetup']

    sampleNoise(system, ctx.config, ctx.schemeConfig, simSetup, args)
    setupFreestream(system, ctx.config, ctx.schemeConfig, simSetup, args)
    setupKolmogorov(system, ctx.config, ctx.schemeConfig, simSetup, args)

    # The sound speed and dt are chosen together: dt follows from the acoustic
    # CFL, so this is what finally fixes config.dt for the run.
    ctx.schemeConfig.fluid.fixedSoundSpeed, ctx.config.dt = setupWeaklyCompressibleTimestep(
        ctx.config, ctx.schemeConfig, system, ctx.param('targetDt'), verbose=ctx.spec.verbose)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    particles = state.state
    fluid = particles.kinds == 0
    velocities = particles.velocities[fluid]
    return {
        'maxVelocity': torch.linalg.norm(velocities, dim=-1).max().detach().cpu().item(),
        'kineticEnergy': (0.5 * particles.masses[fluid]
                          * (velocities ** 2).sum(dim=-1)).sum().detach().cpu().item(),
        'maxDensity': particles.densities[fluid].max().detach().cpu().item(),
        'minDensity': particles.densities[fluid].min().detach().cpu().item(),
    }


def setupPlot(ctx: RunContext, state):
    from ..caseUtils.weaklyCompressiblePlot import setupPlotter

    backend = resolvePlotBackend(ctx)
    try:
        plotter = setupPlotter(state, ctx.scratch['args'], ctx.scratch['simSetup'],
                               ctx.config, ctx.schemeConfig, backend=backend)
    except Exception as exc:
        if backend == 'matplotlib':
            raise
        print(f'the {backend!r} plotting backend failed to start '
              f'({type(exc).__name__}: {exc}); falling back to matplotlib.')
        plotter = setupPlotter(state, ctx.scratch['args'], ctx.scratch['simSetup'],
                               ctx.config, ctx.schemeConfig, backend='matplotlib')
        backend = 'matplotlib'
    ctx.scratch['plotBackend'] = backend
    if ctx.imagePath:
        plotter.export(os.path.join(ctx.imagePath, 'frame_00000.png'), dpi=300)
    openWindow(ctx, plotter)
    return plotter


def updatePlot(ctx: RunContext, state, plotter, step: int) -> None:
    from ..caseUtils.weaklyCompressiblePlot import updatePlot as _update
    _update(plotter, state, ctx.scratch['args'], ctx.scratch['simSetup'],
            ctx.config, ctx.schemeConfig, None)
    if ctx.imagePath:
        plotter.export(os.path.join(ctx.imagePath, f'frame_{step:05d}.png'), dpi=300)
    pumpEvents(plotter)


def extraData(ctx: RunContext, state) -> Dict[str, Any]:
    simSetup = ctx.scratch['simSetup']
    data = {k: v for k, v in ctx.spec.params.items() if not isinstance(v, (list, dict))}
    data.update(
        nx=ctx.spec.nx, L=ctx.spec.L, n_h=ctx.spec.n_h, timeLimit=ctx.spec.tLimit,
        freeSurface=simSetup.freeSurface, dx=simSetup.dx,
        obstacleText=(f"obstacle_{ctx.param('maxExtent'):.4g}_{ctx.param('aoa'):.4g}"
                      f"_{ctx.param('offsetX'):.4g}" if ctx.param('obstacleActive')
                      else 'no_obstacle'),
    )
    return data


dambreakCase = registerCase(Case(
    name='dambreak',
    scheme='deltaSPH',
    description='Dam break with optional obstacle (2D), weakly compressible deltaSPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=extraData,
    defaults=dict(
        caseName='3-dambreak',
        dim=2,
        nx=128,
        L=2.0,
        n_h=4.0,
        kernel='Wendland4',
        integrationScheme='rungeKutta2',
        supportMode='KernelMeanSymmetric',
        tLimit=4.0,
        dt=None,
        adaptiveDt=True,
        cflFactor=0.3,
        minDt=1e-8,
        storeMode='trajectory',
        exportInterval=0.002,
        plotInterval=10,
    ),
    params=dict(
        W=4.0,
        band=5,
        targetDt=0.0005,
        fillRatio=1.0 / 3.0,
        fluidWidth=5 / 2 * 1 / 3,
        semiPeriodic=False,
        fullyPeriodic=False,

        disableGravity=False,
        gravityMagnitude=9.81,
        gravityDirection=[0.0, -1.0],

        obstacleActive=False,
        # `circle` was parser.py's default but is not a preset key any more --
        # generator.py crashes on its own defaults because of it.
        obstacleType='circleMiddle',
        offsetX=3.0 / 4.0,
        aoa=0.0,
        maxExtent=1.0 / 16.0,

        enableFreestream=False,
        forcingWidth=2.0 / 16.0,
        freeStreamVelocity=1.0,

        enableNoise=False,
        octaves=3,
        lacunarity=2,
        persistence=0.5,
        baseFrequency=2,
        kind='perlin',
        seed=45906734,
        noiseAmplitude=1.0,
        bandWidth=16.0,

        enableKolmogorovForcing=False,
        kolmogorovForcingAmplitude=1 / 3,
        kolmogorovForcingWavenumber=2,

        markerSize=4,
        plotWidth=28,
        plotHeight=8,
        plotDensity=False,
    ),
))


if __name__ == '__main__':
    caseMain(dambreakCase)
