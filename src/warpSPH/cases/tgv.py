"""Taylor-Green vortex (2D), incompressible.

The script form of this case was `examples/incompressible/01-tgv-incomp.py`.
Two things that file did are dropped here rather than carried over: it built a
`regions` list that was never passed anywhere, and it imported the local
zero-byte `dfsph.py` / `dfsph_step.py` -- the real step function comes from
`buildScheme`, which the runner already unpacks.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import torch

from ..modules import computeDensities, shuffleParticles
from ..runner import Case, RunContext, caseMain, registerCase
from ..sample.weaklyCompressible import setupBasicWeaklyCompressibleInitialState

__all__ = ['tgvCase', 'analyticDecayRate']


def analyticDecayRate(ctx: RunContext) -> float:
    """Viscous decay rate of the TGV kinetic energy, ``KE(t) = KE(0) e^(-2 nu k^2 t)``."""
    k = ctx.param('k') / 2.0
    return 2.0 * ctx.param('nu') * k ** 2 * 2.0


def configureScheme(ctx: RunContext) -> None:
    # The TGV lives on [0, 2pi]^2, not the symmetric box buildDomainDescription
    # hands back, so the domain is adjusted before any particle is sampled.
    domain = ctx.config.domain
    domain.min = torch.zeros(ctx.spec.dim, device=ctx.device, dtype=ctx.dtype)
    domain.max = torch.ones(ctx.spec.dim, device=ctx.device, dtype=ctx.dtype) * ctx.spec.L

    ctx.schemeConfig.surfaceDetectionConfig.active = ctx.param('freeSurface')
    ctx.schemeConfig.diffusionParams.inviscid = False
    ctx.schemeConfig.diffusionParams.viscidNu = ctx.param('nu')
    ctx.schemeConfig.shiftProperties.active = ctx.param('shifting')


def buildSystem(ctx: RunContext):
    system = setupBasicWeaklyCompressibleInitialState(
        ctx.spec.nx, ctx.config, ctx.schemeConfig, ctx.SimulationState, ctx.SimulationSystem)

    # Normalise the mass so the sampled density lands on rho0 rather than on
    # whatever the regular lattice happens to integrate to.
    rho0 = ctx.param('rho0')
    densities = computeDensities(system.state, ctx.config, ctx.schemeConfig, None)
    system.state.masses = system.state.masses / densities.mean() * rho0
    if ctx.spec.verbose:
        corrected = computeDensities(system.state, ctx.config, ctx.schemeConfig, None)
        print(f'density after mass correction: mean={corrected.mean().item():.6g}, '
              f'min={corrected.min().item():.6g}, max={corrected.max().item():.6g}')
    return system


def initialConditions(ctx: RunContext, system) -> None:
    _relax(ctx, system)

    k = ctx.param('k')
    uMag = ctx.param('uMag')
    kTgv = k / 2.0
    # An even wavenumber puts the vortex centres on the domain boundary; the
    # quarter-period shift moves them back into the interior.
    phase = np.pi / 2 if k % 2 == 0 else 0.0

    positions = system.state.positions
    system.state.velocities[:, 0] = (
        uMag * torch.cos(kTgv * positions[:, 0] + phase) * torch.sin(kTgv * positions[:, 1] + phase))
    system.state.velocities[:, 1] = (
        -uMag * torch.sin(kTgv * positions[:, 0] + phase) * torch.cos(kTgv * positions[:, 1] + phase))


def _relax(ctx: RunContext, system) -> None:
    """Jitter then pressure-relax the lattice, so the IC is not perfectly regular.

    A perfectly regular lattice is an unstable equilibrium for SPH; relaxing it
    first is what keeps the early trajectory free of lattice noise.
    """
    steps = ctx.param('relaxSteps')
    if not steps:
        return

    from warpSPHCore import SupportScheme, buildVerletList
    from ..modules import solveIncompressible

    state = system.initializeNewState()
    state.state.positions = shuffleParticles(state.state, ctx.config, ctx.schemeConfig, 0,
                                             jitterAmount=ctx.param('jitter'))
    state.state.velocities = torch.zeros_like(state.state.velocities)

    adjacency = None
    dt = ctx.param('relaxDt')
    for _ in _maybeProgress(range(steps), ctx.spec.progress, 'relaxing'):
        adjacency = buildVerletList(state.state, ctx.config.domain, verletScale=1.4,
                                    supportMode=SupportScheme.SuperSymmetric,
                                    priorNeighborhood=adjacency, verbose=False)
        state.state.densities = computeDensities(state.state, ctx.config, ctx.schemeConfig, adjacency)
        state.state.pressures[:] = 0.0
        accel, _, _, _ = solveIncompressible(
            particles=state.state, config=ctx.config, schemeConfig=ctx.schemeConfig,
            adjacency=adjacency,
            dvdt=torch.zeros_like(state.state.velocities), dt=dt, verbose=False)
        state.state.positions = state.state.positions + dt * dt * accel

    system.state.positions = state.state.positions.clone()


def _maybeProgress(iterable, enabled, description):
    if not enabled:
        return iterable
    try:
        from tqdm.autonotebook import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=description, leave=False)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    particles = state.state
    kinetic = 0.5 * (particles.masses * (particles.velocities ** 2).sum(dim=-1)).sum()
    return {
        'kineticEnergy': kinetic.detach().cpu().item(),
        'maxVelocity': torch.linalg.norm(particles.velocities, dim=-1).max().detach().cpu().item(),
    }


def setupPlot(ctx: RunContext, state):
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(1, 1, figsize=(7, 6), squeeze=False)
    ctx.scratch['figure'] = (fig, axis)
    updatePlot(ctx, state, (fig, axis), 0)
    return (fig, axis)


def updatePlot(ctx: RunContext, state, handle, step: int) -> None:
    fig, axis = handle
    ax = axis[0][0]
    ax.clear()
    positions = state.state.positions.detach().cpu().numpy()
    speed = torch.linalg.norm(state.state.velocities, dim=-1).detach().cpu().numpy()
    ax.scatter(positions[:, 0], positions[:, 1], c=speed, s=2, cmap='viridis')
    ax.set_aspect('equal')
    ax.set_title(f'TGV  t = {float(state.t):.4f}')
    fig.canvas.draw()
    if ctx.imagePath:
        fig.savefig(os.path.join(ctx.imagePath, f'frame_{step:05d}.png'), dpi=150)


def extraData(ctx: RunContext, state) -> Dict[str, Any]:
    return {k: ctx.param(k) for k in tgvCase.params}


tgvCase = registerCase(Case(
    name='tgv',
    scheme='divergenceFree',
    description='Taylor-Green vortex (2D), divergence-free incompressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=extraData,
    defaults=dict(
        caseName='01-taylorGreenVortex',
        dim=2,
        nx=256,
        L=2 * np.pi,
        n_h=4.0,
        periodic=True,
        kernel='Wendland2',
        integrationScheme='semiImplicitEuler',
        supportMode='KernelMeanSymmetric',
        tLimit=2.0,
        dt=1e-3,
        adaptiveDt=True,
        cflFactor=0.3,
        minDt=1e-8,
    ),
    params=dict(
        rho0=1.0,
        nu=0.01,
        k=2,
        uMag=1.0,
        freeSurface=False,
        shifting=False,
        relaxSteps=32,
        relaxDt=1e-3,
        jitter=0.01,
    ),
))


if __name__ == '__main__':
    caseMain(tgvCase)
