"""What every weakly compressible example case does the same way.

The `examples/weaklyCompressible/*.ipynb` notebooks all open with the same four
moves: widen the domain by `band` particle layers, describe the fluid and the
walls as SDF regions, sample them, and then pick the sound speed and `dt`
*together* from a target timestep. Only the SDFs and the initial velocity field
differ between them.

That common part is here. A case module supplies :func:`regions`-worth of SDFs
and an initial velocity field, and gets the rest.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from ..configurations.moduleConfigurations.gravity import GravityType
from ..configurations.region import BCType, RegionType
from ..initializers import initializeWeaklyCompressibleSimulation
from ..modules import setupWeaklyCompressibleTimestep
from ..regions import buildRegion, filterRegion, sampleDomainSDF
from ..runner import RunContext, resolveEnum
from ..utils import buildDomainDescription
from ..sampling import getSDF, operatorDict, sampleSDF
from .plotting import Field

__all__ = [
    'WEAKLY_COMPRESSIBLE_DEFAULTS', 'WEAKLY_COMPRESSIBLE_PARAMS',
    'configureWeaklyCompressible', 'domainFluidSdf', 'domainBoundarySdf', 'shapeSdf',
    'buildRegionSystem', 'fluidRegion', 'boundaryRegion',
    'setupTimestep', 'weaklyCompressibleDiagnostics',
    'VELOCITY_DENSITY_FIELDS', 'paramExtraData',
]


#: `CaseSpec` fields shared by the weakly compressible examples.
WEAKLY_COMPRESSIBLE_DEFAULTS = dict(
    dim=2,
    kernel='Wendland4',
    integrationScheme='rungeKutta2',
    supportMode='KernelMeanSymmetric',
    gradientMode='Difference',
    laplacianMode='Brookshaw',
    samplingScheme='regular',
    periodic=True,
    n_h=4.0,
    # dt is set by `setupTimestep` from `targetDt` and the sound speed.
    dt=None,
    adaptiveDt=True,
    cflFactor=0.3,
    minDt=1e-8,
    plotInterval=60,
    storeInterval=500,
)

#: Case parameters shared by the weakly compressible examples.
WEAKLY_COMPRESSIBLE_PARAMS = dict(
    rho0=1.0,
    targetDt=0.0005,
    band=0,
    freeSurface=False,
    inviscid=True,
    nu=0.0,
    markerSize=4,
)

#: The two panels almost every weakly compressible notebook plotted.
VELOCITY_DENSITY_FIELDS = [
    Field('velocities', 'velocities', colorMap='viridis', mapping='L2Norm'),
    Field('densities', 'densities', colorMap='RdBu', colorMapKind='diverging',
          flip=True, midPoint=1.0, vMin=0.99, vMax=1.01),
]


def configureWeaklyCompressible(ctx: RunContext) -> None:
    """Domain, resolution and the scheme knobs the examples share.

    The simulated box is `band` particle layers wider than the *interior*
    domain on every side; the interior is what the walls are cut from, and it
    is stashed on `ctx.scratch` because the boundary SDFs need it.
    """
    dx = ctx.spec.L / ctx.spec.nx
    band = ctx.param('band')

    domain = buildDomainDescription(ctx.spec.L + dx * band * 2, ctx.spec.dim,
                                    ctx.spec.periodic, ctx.device, ctx.dtype)
    interior = buildDomainDescription(ctx.spec.L, ctx.spec.dim, False,
                                      ctx.device, ctx.dtype)
    ctx.config.domain = domain
    ctx.config.dx = dx
    ctx.config.nx = ctx.spec.nx + band * 2
    ctx.scratch['interiorDomain'] = interior

    schemeConfig = ctx.schemeConfig
    schemeConfig.surfaceDetectionConfig.active = ctx.param('freeSurface')
    schemeConfig.diffusionParams.inviscid = ctx.param('inviscid')
    if not ctx.param('inviscid'):
        schemeConfig.diffusionParams.viscidNu = ctx.param('nu')

    if ctx.param('gravity', False):
        schemeConfig.gravityConfig.active = True
        schemeConfig.gravityConfig.type = resolveEnum(GravityType,
                                                      ctx.param('gravityType'))
        schemeConfig.gravityConfig.magnitude = ctx.param('gravityMagnitude')
        schemeConfig.gravityConfig.origin = ctx.param('gravityDirection')
    else:
        schemeConfig.gravityConfig.active = False


# -- SDF helpers -------------------------------------------------------------
# These are the three shapes the notebooks wrote inline, once each per file, as
# multi-line lambdas over `getSDF`/`operatorDict`.

def domainFluidSdf(ctx: RunContext) -> Callable:
    """Fluid filling the whole (band-widened) domain."""
    domain = ctx.config.domain
    return lambda x: sampleDomainSDF(x, domain, invert=True)


def domainBoundarySdf(ctx: RunContext) -> Callable:
    """Walls: everything outside the interior domain."""
    interior = ctx.scratch['interiorDomain']
    return lambda x: sampleDomainSDF(x, interior, invert=False)


def shapeSdf(name: str, size, offset=None, invert: bool = False) -> Callable:
    """One of the built-in implicit shapes, optionally translated.

    `size` is whatever that shape's function takes -- a radius for `circle`,
    half-extents for `box`.
    """
    def sdf(points):
        device = points.device
        shape = lambda x: getSDF(name)['function'](x, torch.as_tensor(size).to(device))
        if offset is not None:
            shape = operatorDict['translate'](shape, torch.as_tensor(offset).to(device))
        return sampleSDF(points, shape, invert=invert)
    return sdf


def buildRegionSystem(ctx: RunContext, regions: Sequence) -> Any:
    """Filter overlapping regions, register them, and sample the particles.

    `filterRegion` is what stops two touching fluid bodies from sampling
    particles on top of each other, so it has to run before the system is
    initialised, not after.
    """
    regions = list(regions)
    for region in regions:
        region = filterRegion(region, regions)
    ctx.config.regions = ctx.schemeConfig.regions = regions
    ctx.scratch['regions'] = regions

    return initializeWeaklyCompressibleSimulation(
        regions, ctx.config, ctx.schemeConfig,
        ctx.SimulationSystem, ctx.SimulationState, verbose=ctx.spec.verbose)


def fluidRegion(ctx: RunContext, sdf: Callable, **kwargs):
    return buildRegion(ctx.config, ctx.schemeConfig, sdf, RegionType.Fluid,
                       initialConditions={}, **kwargs)


def boundaryRegion(ctx: RunContext, sdf: Callable, kind: BCType = BCType.freeSlip, **kwargs):
    return buildRegion(ctx.config, ctx.schemeConfig, sdf, RegionType.Boundary,
                       initialConditions={}, kind=kind, **kwargs)


def setupTimestep(ctx: RunContext, system) -> None:
    """Pick the sound speed and `dt` together, from `targetDt`.

    Weakly compressible SPH is free to choose its own stiffness, so rather than
    a sound speed being given and dt following, the notebooks fix the timestep
    they want and let the sound speed follow from the acoustic CFL. This is the
    call that finally sets `config.dt` for the run.
    """
    ctx.schemeConfig.fluid.fixedSoundSpeed, ctx.config.dt = setupWeaklyCompressibleTimestep(
        ctx.config, ctx.schemeConfig, system, ctx.param('targetDt'),
        verbose=ctx.spec.verbose)


def weaklyCompressibleDiagnostics(ctx: RunContext, state) -> Dict[str, float]:
    """Kinetic energy, peak speed and the density bounds, over fluid only.

    Density bounds are the weakly compressible health check: the whole scheme
    rests on the fluid staying within about a percent of `rho0`.
    """
    particles = state.state
    fluid = particles.kinds == 0 if hasattr(particles, 'kinds') else slice(None)
    velocities = particles.velocities[fluid]
    densities = particles.densities[fluid]
    return {
        'kineticEnergy': (0.5 * particles.masses[fluid]
                          * (velocities ** 2).sum(dim=-1)).sum().detach().cpu().item(),
        'maxVelocity': torch.linalg.norm(velocities, dim=-1).max().detach().cpu().item(),
        'maxDensity': densities.max().detach().cpu().item(),
        'minDensity': densities.min().detach().cpu().item(),
    }


def paramExtraData(ctx: RunContext, state) -> Dict[str, Any]:
    """Record the case's scalar parameters on every exported frame."""
    data = {k: v for k, v in ctx.spec.params.items()
            if not isinstance(v, (list, dict))}
    data.update(nx=ctx.spec.nx, L=ctx.spec.L, n_h=ctx.spec.n_h,
                dx=ctx.config.dx, timeLimit=ctx.spec.tLimit)
    return data
