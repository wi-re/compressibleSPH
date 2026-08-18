"""Convergence check for implicit particle shifting
(`warpSPH.modules.shifting.implicitShifting.computeImplicitShift`, dispatched
via `modules/shifting/wrapper.py::solveShifting` when
`shiftProperties.scheme == ShiftingScheme.implicit`): starting from a
randomly jittered lattice in a periodic domain -- the same "relax toward a
glass configuration" setup `sample.optimal.sampleOptimal` uses for the
explicit delta-shift -- both the explicit delta-SPH shift and the new
implicit shift should drive the particle distribution toward spatial
uniformity, tracked via the relative SPH-density deviation from `rho0` (the
quantity the implicit solve's RHS directly targets).
"""

import pytest
import torch

from warpSPH.utils import buildDomainDescription
from warpSPH.configurations.simulationConfig import buildConfig
from warpSPH.configurations.weaklyCompressible import WeaklyCompressibleSPHConfig
from warpSPH.configurations.moduleConfigurations.shifting import ShiftingScheme
from warpSPH.sample.regular import sampleRegularParticles
from warpSPH.modules.density import computeDensities
from warpSPH.modules.shifting import solveShifting
from warpSPHCore import ParticleState, SupportScheme, buildVerletList


def _jitteredLatticeParticles(nx, dim, L, device, dtype, jitter, seed):
    domain = buildDomainDescription(l=L, dim=dim, periodic=True, device=device, dtype=dtype)
    config, _integrator = buildConfig(dim=dim, nx=nx, domain=domain, device=device,
                                       dtype=dtype, dx=L / nx, cflFactor=0.3)

    template = sampleRegularParticles(nx=nx, domain=domain, targetNeighbors=config.targetNeighbors)
    n = template.positions.shape[0]
    particleDx = template.masses.pow(1 / dim).mean().item()

    gen = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(template.positions.shape, generator=gen).to(device=device, dtype=dtype)
    positions = template.positions + jitter * particleDx * noise

    state = ParticleState(
        positions=positions,
        supports=template.supports,
        masses=template.masses,
        kinds=torch.zeros(n, device=device, dtype=torch.int32),
        densities=torch.ones(n, device=device, dtype=dtype),
    )
    # computeDeltaShift (delta.py) unconditionally reads currentState.velocities
    # for its Mach-number estimate, even with computeMach disabled.
    state.velocities = torch.zeros_like(state.positions)
    return state, config


def _relativeDensityDeviation(state, config, schemeConfig, rho0):
    adjacency = buildVerletList(state, config.domain, config.verletScale, SupportScheme.SuperSymmetric, None)
    densities = computeDensities(state, config, schemeConfig, adjacency)
    return ((densities - rho0) / rho0), densities


def _relaxAndTrack(state, config, schemeConfig, rho0, outerIters):
    history = []
    rel, densities = _relativeDensityDeviation(state, config, schemeConfig, rho0)
    state.densities = densities
    history.append(rel.std().item())

    adjacency = None
    for _ in range(outerIters):
        dx = solveShifting(state, config, schemeConfig, adjacency, dt=1.0)
        state.positions = state.positions + dx
        rel, densities = _relativeDensityDeviation(state, config, schemeConfig, rho0)
        state.densities = densities
        history.append(rel.std().item())
    return history


@pytest.mark.parametrize('scheme', [ShiftingScheme.deltaSPH, ShiftingScheme.implicit])
def test_shiftingConvergesToUniformDensity(scheme):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    dim, nx, L = 2, 16, 1.0

    state, config = _jitteredLatticeParticles(
        nx=nx, dim=dim, L=L, device=device, dtype=dtype, jitter=0.1, seed=1234,
    )

    schemeConfig = WeaklyCompressibleSPHConfig()
    schemeConfig.shiftProperties.scheme = scheme
    schemeConfig.shiftProperties.iterations = 1
    schemeConfig.shiftProperties.active = True
    # SurfaceDetectionConfig.active defaults to True, but this lattice is
    # fully periodic with no real free surface -- solveShifting's ColorField
    # heuristic still fires false positives on it as particles rearrange,
    # and its post-hoc shift zeroing/projection for "surface" particles
    # destabilizes the implicit Newton solve (confirmed: with detection left
    # on, ShiftingScheme.implicit's density std blows up from ~0.008 back up
    # to ~0.25 partway through the 25-iteration relaxation; deltaSPH's
    # smaller clamped per-step corrections happen to tolerate the same
    # interference). Off, both schemes converge monotonically.
    schemeConfig.surfaceDetectionConfig.active = False
    rho0 = schemeConfig.fluid.restDensity

    history = _relaxAndTrack(state, config, schemeConfig, rho0, outerIters=25)

    # deltaSPH's CFL-limited per-step correction converges much more slowly
    # than the implicit Newton solve, so a shared bound has to accommodate
    # both -- this only checks that shifting made real, non-trivial progress
    # toward a uniform distribution, not that it fully converged in 25 steps.
    assert history[-1] < 0.6 * history[0], (
        f'{scheme}: relative density std only went from {history[0]:.4f} to {history[-1]:.4f} '
        f'over {len(history) - 1} steps (history={history})')
    assert min(history[1:]) <= history[0], (
        f'{scheme}: never improved on the initial configuration (history={history})')


if __name__ == '__main__':
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')

    for scheme in (ShiftingScheme.deltaSPH, ShiftingScheme.implicit):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        state, config = _jitteredLatticeParticles(
            nx=16, dim=2, L=1.0, device=device, dtype=torch.float32, jitter=0.1, seed=1234,
        )
        schemeConfig = WeaklyCompressibleSPHConfig()
        schemeConfig.shiftProperties.scheme = scheme
        schemeConfig.shiftProperties.iterations = 1
        schemeConfig.shiftProperties.active = True
        schemeConfig.surfaceDetectionConfig.active = False
        rho0 = schemeConfig.fluid.restDensity

        history = _relaxAndTrack(state, config, schemeConfig, rho0, outerIters=25)
        print(f'{scheme.name}: {[f"{h:.4f}" for h in history]}')
