"""Short runs of each converted case, asserting a physical invariant.

Deliberately tiny -- 20 steps at a coarse resolution -- so the suite stays
usable as a pre-commit check. Each assertion is a property the scheme is
supposed to have, not a golden number, so they survive refactors and fail on
real regressions.
"""

import io
import contextlib

import numpy as np
import pytest

from warpSPH.runner import run

STEPS = 20


def _run(case, **overrides):
    """Run quietly; the schemes print per-step solver diagnostics."""
    with contextlib.redirect_stdout(io.StringIO()):
        return run(case, progress=False, nSteps=STEPS, **overrides)


@pytest.fixture(scope='module')
def sodResult():
    from warpSPH.cases.sod import sodCase
    return _run(sodCase, nx=200)


@pytest.fixture(scope='module')
def tgvResult():
    from warpSPH.cases.tgv import tgvCase
    return _run(tgvCase, nx=32)


@pytest.fixture(scope='module')
def dambreakResult():
    from warpSPH.cases.dambreak import dambreakCase
    return _run(dambreakCase, nx=48)


# --- Sod: compressible, energy conserving -----------------------------------

def test_sodConservesTotalEnergy(sodResult):
    """CompSPH is an energy-conserving discretisation: the kinetic/thermal
    exchange across the shock must not change the total."""
    energy = sodResult.series('totalEnergy')
    drift = abs(energy[-1] - energy[0]) / abs(energy[0])
    assert drift < 1e-5, f'total energy drifted by {drift:.3e}'


def test_sodConvertsThermalEnergyIntoMotion(sodResult):
    """The tube starts at rest; the pressure jump has to set it moving."""
    kinetic = sodResult.series('kineticEnergy')
    assert kinetic[0] == pytest.approx(0.0, abs=1e-12)
    assert kinetic[-1] > 0.0
    assert sodResult.series('thermalEnergy')[-1] < sodResult.series('thermalEnergy')[0]


def test_sodDoesNotDiverge(sodResult):
    assert not sodResult.diverged
    assert len(sodResult.trajectory) == STEPS + 1


# --- Sod in 2D and 3D: the same tube, sampled at equal mass ------------------

#: `nx` here is the dense side's count across its own half of the domain, so
#: these are ~500 (2D) and ~4000 (3D) particles -- the 3D one is the most
#: expensive test in the suite even at that size. `transverseSpacings` is left
#: at its default: it is a multiple of the particle spacing, so the minimum the
#: sampler will accept (~16 in 3D) does not fall with `nx`, and trimming it to
#: make the test cheaper only trips the periodic-image guard.
_SOD_ND = {'sod2d': dict(nx=20), 'sod3d': dict(nx=8)}


@pytest.fixture(scope='module', params=sorted(_SOD_ND))
def sodNDResult(request):
    from warpSPH.cases import sodND
    case = {'sod2d': sodND.sod2dCase, 'sod3d': sodND.sod3dCase}[request.param]
    return request.param, _run(case, **_SOD_ND[request.param])


def test_sodNDConservesTotalEnergyAndMoves(sodNDResult):
    """The 1D assertions, in the dimension the case actually runs in."""
    name, result = sodNDResult
    assert not result.diverged, f'{name} diverged'
    energy = result.series('totalEnergy')
    drift = abs(energy[-1] - energy[0]) / abs(energy[0])
    assert drift < 1e-5, f'{name} total energy drifted by {drift:.3e}'
    kinetic = result.series('kineticEnergy')
    assert kinetic[0] == pytest.approx(0.0, abs=1e-12)
    assert kinetic[-1] > 0.0, f'{name} never started moving'
    assert result.series('thermalEnergy')[-1] < result.series('thermalEnergy')[0]


def test_sodNDStaysUniformAcrossTheSlab(sodNDResult):
    """The transverse directions are periodic and the solution does not depend
    on them, so the spread of density at a given x measures how much the
    sampling and the periodic wrap are corrupting a 1D answer."""
    name, result = sodNDResult
    state = result.state.state
    x = state.positions[:, 0].detach().cpu().numpy()
    rho = state.densities.detach().cpu().numpy()
    # Away from the waves, where the answer is still exactly the initial state.
    quiet = np.abs(x) < 0.15
    assert quiet.sum() > 10
    spread = rho[quiet].std() / rho[quiet].mean()
    assert spread < 0.02, f'{name} density varies by {spread:.2%} across the slab'


@pytest.mark.parametrize('dim,nx', [(1, 100), (1, 800), (2, 20), (2, 100), (3, 10), (3, 40)])
def test_sodSamplingMatchesParticleMasses(dim, nx):
    """The point of `sodND`'s sampler. Equal *spacing* would leave the dense
    side's particles `rho_l/rho_r = 4` times heavier (a 75% mismatch); equal
    mass is what it is for, and 3D cannot be exact -- `4**(1/3)` is irrational
    -- so it is allowed a little slack, but not much."""
    from warpSPH.caseUtils import sodInitialState, sodSampling
    sampling = sodSampling(nx, dim, 2.0, 20 if dim > 1 else 1,
                           sodInitialState(p=1.0, rho=1.0, v=0.0),
                           sodInitialState(p=0.1795, rho=0.25, v=0.0))
    assert abs(sampling.massRatio - 1) < 0.05, (
        f'dim={dim} nx={nx}: masses differ by {abs(sampling.massRatio - 1):.2%}')
    assert sampling.anisotropy <= 1.25, (
        f'dim={dim} nx={nx}: cells stretched {sampling.anisotropy:.3f}:1')


def test_sodSamplingAgreesWithThe1DBuilder():
    """`sodND` at dim=1 must reproduce `buildSod1D`'s lattice, or the 2D/3D
    cases are not extruding the case the 1D one actually runs."""
    from warpSPH.caseUtils import sodInitialState, sodSampling
    left = sodInitialState(p=1.0, rho=1.0, v=0.0)
    right = sodInitialState(p=0.1795, rho=0.25, v=0.0)
    for nx in (100, 800):
        sampling = sodSampling(nx, 1, 2.0, 1, left, right)
        # buildSod1D: dx = (L/2)/nx on the left, samplingRatio=4 coarser on the
        # right, giving nx//4 particles over the same extent.
        assert sampling.dense[2] == pytest.approx(1.0 / nx)
        assert sampling.light[0] == nx // 4
        assert sampling.light[2] == pytest.approx(4.0 / nx)
        assert sampling.massRatio == pytest.approx(1.0)


def test_sodNDKeepsTheLightStateOutOfTheDenseBlock():
    """`buildSod1D` samples the light state as one block centred on the origin
    and pushes its halves outward, which leaves a particle at exactly x=0 --
    inside the *dense* state, carrying the light state's mass -- whenever the
    light count is odd (`nx=100` does it; the `nx=800` default does not).
    `sodND` lays the same state out as a wrapped periodic interval instead, so
    it cannot happen at any `nx`. Both are checked here: the 1D builder's
    behaviour is recorded rather than fixed, since the backprop notebook's
    numbers are tied to its exact output.
    """
    import io
    import contextlib

    import torch

    from warpSPH.caseUtils import buildSodND, sodInitialState
    from warpSPH.cases.sod import sodCase
    from warpSPH.runner import CaseSpec, buildContext

    def build(nx, dim):
        spec = CaseSpec(caseName=sodCase.name, scheme=sodCase.scheme,
                        params=dict(sodCase.params)).merged(**sodCase.defaults)
        spec = spec.merged(nx=nx, dim=dim, plot=False, store=False, quiet=True)
        ctx = buildContext(sodCase, spec)
        sodCase.configureScheme(ctx)
        with contextlib.redirect_stdout(io.StringIO()):
            oneD = sodCase.buildSystem(ctx).state
            nd = buildSodND(
                ctx.SimulationSystem, ctx.SimulationState,
                sodInitialState(p=1.0, rho=1.0, v=0.0),
                sodInitialState(p=0.1795, rho=0.25, v=0.0),
                ctx.param('gamma'), ctx.config, transverseSpacings=1, verbose=False).state
        return oneD, nd

    def strandedLightParticles(state):
        return int(((state.materials == 1) & (state.positions[:, 0].abs() < 0.5 - 1e-6)).sum())

    oneD, nd = build(nx=100, dim=1)
    assert strandedLightParticles(oneD) == 1, 'buildSod1D no longer strands one at x=0'
    assert strandedLightParticles(nd) == 0
    # Same particle count and the same set of masses, stray particle aside.
    assert nd.positions.shape[0] == oneD.positions.shape[0]
    assert torch.allclose(nd.masses.sort().values, oneD.masses.sort().values)


def test_adaptiveSupportDoesNotDependOnWhatRanBefore():
    """Owen's psi lookup table is sliced by dimension and cached, so building a
    2D case in between two 1D builds must not change the 1D answer.

    It used to: the cache was a single unkeyed global, so the first dimension a
    *process* touched won, and everything after it got supports relaxed against
    the wrong table -- silently, since a support radius has no obviously wrong
    value. Invisible until a case ran in two dimensions in one process, which
    nothing did before `sod3d`.
    """
    import io
    import contextlib

    from warpSPH.cases import sodND  # noqa: F401 - registers sod2d/sod3d
    from warpSPH.runner import CaseSpec, buildContext, getCase

    def supports(name, **overrides):
        case = getCase(name)
        spec = CaseSpec(caseName=case.name, scheme=case.scheme,
                        params=dict(case.params)).merged(**case.defaults)
        spec = spec.merged(plot=False, store=False, quiet=True, **overrides)
        ctx = buildContext(case, spec)
        case.configureScheme(ctx)
        with contextlib.redirect_stdout(io.StringIO()):
            return case.buildSystem(ctx).state.supports.max().item()

    before = supports('sod', nx=100)
    supports('sod2d', nx=20)
    after = supports('sod', nx=100)
    assert after == pytest.approx(before, rel=1e-9), (
        f'a 2D build in between changed the 1D supports: {before} -> {after}')


def test_sodNDRejectsASlabNarrowerThanItsKernel():
    """A slab under twice the support radius lets particles interact with their
    own periodic images, and nothing downstream would notice -- so the sampler
    has to refuse rather than return a quietly wrong initial condition."""
    from warpSPH.cases import sodND
    with pytest.raises(ValueError, match='periodic images'):
        _run(sodND.sod2dCase, nx=20, params=dict(transverseSpacings=4))


# --- the three compressible solvers, as comparison runs ----------------------

#: Total-energy drift each solver is allowed over 20 steps. CompSPH is
#: energy-conserving by construction and measures exactly 0; CRKSPH is
#: conservative to round-off. Monaghan is *not* an energy-conserving
#: discretisation -- its artificial viscosity and conductivity are dissipative
#: by design -- so it gets a loose bound that still catches a solver that has
#: stopped integrating anything.
_ENERGY_DRIFT = {'CompSPH': 1e-5, 'CRKSPH': 1e-4, 'Monaghan': 5e-3}


@pytest.fixture(scope='module', params=sorted(_ENERGY_DRIFT))
def compressibleSchemeResult(request):
    """The Sod tube under each solver `--scheme` can select.

    These exist because **Monaghan was broken and nothing noticed**: it called
    all three boundary-condition helpers with their pre-`t` argument list, and
    `computeMomentumConsistent` with a `supportScheme=` keyword the function no
    longer takes. Every compressible case defaults to CRKSPH or CompSPH, so the
    only way to reach it was `--scheme Monaghan`, which no test did. Any solver
    reachable from the command line is now exercised here.
    """
    from warpSPH.cases.sod import sodCase
    return request.param, _run(sodCase, nx=200, scheme=request.param)


def test_everyCompressibleSolverRuns(compressibleSchemeResult):
    scheme, result = compressibleSchemeResult
    assert not result.diverged, f'{scheme} diverged'
    assert len(result.trajectory) == STEPS + 1


def test_everyCompressibleSolverConvertsThermalEnergyIntoMotion(compressibleSchemeResult):
    """The shared physics: the tube starts at rest and the pressure jump moves
    it, whichever discretisation is doing the integrating."""
    scheme, result = compressibleSchemeResult
    kinetic = result.series('kineticEnergy')
    thermal = result.series('thermalEnergy')
    assert kinetic[0] == pytest.approx(0.0, abs=1e-12)
    assert kinetic[-1] > 0.0, f'{scheme} never started moving'
    assert thermal[-1] < thermal[0], f'{scheme} did not convert thermal energy'


def test_everyCompressibleSolverKeepsEnergyDriftInBounds(compressibleSchemeResult):
    scheme, result = compressibleSchemeResult
    energy = result.series('totalEnergy')
    drift = abs(energy[-1] - energy[0]) / abs(energy[0])
    assert drift < _ENERGY_DRIFT[scheme], (
        f'{scheme} total energy drifted by {drift:.3e}')


def test_theSolverIsSelectedByTheSchemeFlag():
    """`--scheme` has to reach `buildScheme`, or a comparison run would
    silently compare a scheme against itself."""
    from warpSPH.cases.sod import sodCase
    from warpSPH.runner import buildContext
    from warpSPH.runner.caseSpec import CaseSpec

    expected = {'CompSPH': 'compSPH_step', 'CRKSPH': 'crkSPH_step',
                'Monaghan': 'compressibleSPH_Monaghan'}
    for scheme, stepFunction in expected.items():
        spec = CaseSpec(caseName=sodCase.name, scheme=scheme,
                        params=dict(sodCase.params)).merged(**sodCase.defaults)
        ctx = buildContext(sodCase, spec)
        assert ctx.stepFunction.__name__ == stepFunction, scheme


# --- TGV: incompressible, viscously decaying --------------------------------

def test_tgvKineticEnergyDecaysAtRoughlyTheAnalyticRate(tgvResult):
    """`KE(t) = KE(0) exp(-4 nu k^2 t)` for the Taylor-Green vortex.

    The measured rate sits near 0.55-0.6x the analytic one and is *stable under
    refinement* (0.605 at nx=32/20 steps, 0.564 at nx=32/50, 0.550 at
    nx=64/200), so it is not discretisation error.

    It is the Monaghan switch in the diffusion operator: viscosity is
    deactivated for particle pairs that are separating, so only the approaching
    half of the pairs dissipates at any instant and the effective viscosity is
    roughly half the prescribed nu. This is expected SPH behaviour, not a bug.
    Disabling the switch does recover the analytic decay rate, at the cost of
    stability elsewhere in the simulation, so it stays on.

    The band is therefore wide on purpose: it catches viscosity being
    disconnected (rate -> 0) or mis-scaled, without pretending the ~0.55 factor
    is an error to be driven out.
    """
    from warpSPH.cases.tgv import analyticDecayRate

    kinetic = tgvResult.series('kineticEnergy')
    time = tgvResult.series('t')
    measured = -np.polyfit(time, np.log(kinetic), 1)[0]
    analytic = analyticDecayRate(tgvResult.ctx)

    assert measured / analytic == pytest.approx(0.6, rel=0.45)


def test_tgvKineticEnergyIsMonotoneDecreasing(tgvResult):
    kinetic = tgvResult.series('kineticEnergy')
    assert np.all(np.diff(kinetic) < 0)


def test_tgvDoesNotDiverge(tgvResult):
    assert not tgvResult.diverged


# --- Dam break: weakly compressible, gravity driven -------------------------

def test_dambreakStaysWeaklyCompressible(dambreakResult):
    """The defining property of the scheme: density stays within ~1% of rho0."""
    maxDensity = max(row['maxDensity'] for row in dambreakResult.trajectory)
    minDensity = min(row['minDensity'] for row in dambreakResult.trajectory)
    assert 0.99 < minDensity <= maxDensity < 1.01


def test_dambreakGravityDoesWorkOnTheFluid(dambreakResult):
    """The column starts at rest and is released; kinetic energy has to grow."""
    kinetic = dambreakResult.series('kineticEnergy')
    assert kinetic[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(kinetic) > 0)


def test_dambreakDoesNotDiverge(dambreakResult):
    assert not dambreakResult.diverged
