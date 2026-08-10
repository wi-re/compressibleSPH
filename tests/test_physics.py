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
