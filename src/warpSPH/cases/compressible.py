"""What every compressible example case does the same way.

The fifteen `examples/compressible/*.ipynb` notebooks share a setup block
verbatim: CRKSPH, the B7 kernel, `gamma`/`rho0` onto the scheme config, the
viscosity switch and the Owen adaptive-support scheme. They also share their
diagnostics -- kinetic, thermal and total energy, with total energy the
conserved quantity the runs are judged on.

That block lives here once, so each case module is only its own geometry,
initial condition and plot.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from ..enumTypes import AdaptiveSupportScheme, ViscositySwitch
from ..modules.timestep.compressible import computeTimestep
from ..runner import RunContext, resolveEnum

__all__ = ['COMPRESSIBLE_DEFAULTS', 'COMPRESSIBLE_PARAMS', 'configureCompressible',
           'compressibleDiagnostics', 'compressibleTimestep', 'paramExtraData']


#: `CaseSpec` fields shared by the compressible examples. A case merges these
#: into its own `defaults` and overrides what it needs (`nx`, `dim`, `tLimit`).
COMPRESSIBLE_DEFAULTS = dict(
    kernel='B7',
    integrationScheme='rungeKutta2',
    supportMode='KernelMeanSymmetric',
    gradientMode='Difference',
    laplacianMode='Brookshaw',
    samplingScheme='regular',
    periodic=True,
    n_h=4.0,
    # Left unset: every compressible sampler ends by calling `computeTimestep`,
    # so the CFL-derived value is what the run actually starts from.
    dt=None,
    adaptiveDt=True,
    cflFactor=0.3,
    minDt=1e-8,
    plotInterval=25,
    storeInterval=500,
)

#: Case parameters shared by the compressible examples.
COMPRESSIBLE_PARAMS = dict(
    gamma=5 / 3,
    rho0=1.0,
    viscositySwitch='NoneSwitch',
    adaptiveSupportScheme='Owen',
    adaptiveSupportCorrections=False,
    markerSize=2,
)


def configureCompressible(ctx: RunContext) -> None:
    """Stamp the shared scheme settings onto `ctx.schemeConfig`."""
    schemeConfig = ctx.schemeConfig
    schemeConfig.gamma = ctx.param('gamma')
    schemeConfig.rho0 = ctx.param('rho0')
    schemeConfig.viscositySwitchParams.scheme = resolveEnum(
        ViscositySwitch, ctx.param('viscositySwitch'))
    schemeConfig.adaptiveSupportScheme = resolveEnum(
        AdaptiveSupportScheme, ctx.param('adaptiveSupportScheme'))
    schemeConfig.adaptiveSupportCorrections = ctx.param('adaptiveSupportCorrections')


def compressibleDiagnostics(ctx: RunContext, state) -> Dict[str, float]:
    """Kinetic, thermal and total energy -- total energy is the conserved one."""
    particles = state.state
    kinetic = 0.5 * (torch.linalg.norm(particles.velocities, dim=-1) ** 2
                     * particles.masses).sum()
    thermal = (particles.internalEnergies * particles.masses).sum()
    return {
        'kineticEnergy': kinetic.detach().cpu().item(),
        'thermalEnergy': thermal.detach().cpu().item(),
        'totalEnergy': (kinetic + thermal).detach().cpu().item(),
    }


def compressibleTimestep(ctx: RunContext, state) -> "float | torch.Tensor":
    """The acoustic-CFL `dt`, recomputed from the current state.

    Attach this as a case's `timestep` hook to get the notebooks'
    ``while t < tLimit`` behaviour: dt tracks the sound speed as the shock
    develops instead of staying at whatever the initial state implied.
    """
    return computeTimestep(state, ctx.config, ctx.schemeConfig, dt=ctx.config.dt)


def paramExtraData(ctx: RunContext, state) -> Dict[str, Any]:
    """Record the case's own parameters on every exported frame.

    Lists and dicts are dropped: the HDF5 attribute writer only takes scalars
    and strings, and a nested region description is not what a frame needs.
    """
    return {k: v for k, v in ctx.spec.params.items()
            if not isinstance(v, (list, dict))}
