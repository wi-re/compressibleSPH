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


# --- TGV: incompressible, viscously decaying --------------------------------

def test_tgvKineticEnergyDecaysAtRoughlyTheAnalyticRate(tgvResult):
    """`KE(t) = KE(0) exp(-4 nu k^2 t)` for the Taylor-Green vortex.

    The band is wide because at nx=32 the viscous operator is badly
    under-resolved -- measured rates sit near 0.55-0.6x analytic and are stable
    under refinement, so this catches viscosity being disconnected (rate -> 0)
    or mis-scaled, not discretisation error.
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
