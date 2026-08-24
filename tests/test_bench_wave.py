"""Smoke + consistency tests for `benchmarks/` (the wave-case integrator
benchmark suite).

Kept small on purpose (nx=16, a handful of steps): what is pinned here is the
suite's own instrumentation -- RHS counting, the internal-solver record,
multistep history threading, run reproducibility from one built system, and
that the accuracy machinery actually measures convergence -- not performance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.common import SCHEMES, getSchemes, runScheme, buildWaveCase, relL2
from benchmarks.common.schemes import MULTISTEP_SCHEMES
from warpSPHIntegrators import NonlinearSolver, getIntegrationEnum, getIntegrator


@pytest.fixture(scope='module')
def case():
    """One built case (nx=16, the case's own CFL dt) shared by all tests.

    The fixture also doubles as the proof that `runScheme` may be called
    repeatedly against one built `system` (see
    `testReuseBuiltSystemReproducible`)."""
    ctx, system, buildSeconds = buildWaveCase(nx=16)
    return ctx, system, float(ctx.config.dt)


def testSchemeRegistryResolves():
    """Every registry key resolves through the same name path the frontend
    uses, carries its true order, and (implicit only) a buildable solver."""
    assert len(SCHEMES) >= 60
    for key, spec in SCHEMES.items():
        getIntegrationEnum(spec.integrationScheme)  # raises on drift
        assert spec.order == getIntegrator(spec.integrationScheme).order, key
        assert spec.kind in ('explicit', 'implicit'), key
        if spec.kind == 'explicit':
            assert spec.makeSolver is None, key
        else:
            solver = spec.makeSolver()
            assert isinstance(solver, NonlinearSolver), key
    # the notebook's configuration must exist under a stable key
    assert 'sdirk2_jfnk_jvp_1e-6' in SCHEMES
    assert 'sdirk2_jfnk_fd_1e-6' in SCHEMES
    assert [s.key for s in getSchemes(['sdirk2_jfnk_jvp_1e-6'], [])] \
        == ['sdirk2_jfnk_jvp_1e-6']
    # multistep names agree with the registry's own bookkeeping
    for name in MULTISTEP_SCHEMES:
        assert name in SCHEMES


def testRK4CountsFourRhsPerStep(case):
    ctx, system, dt = case
    rec = runScheme(ctx, system, getSchemes(['rk4'], None)[0], 3, dt,
                    warmup=1, keepFields=False)
    assert not rec.diverged
    assert rec.fEvals == 4 * 4          # 1 warmup + 3 timed steps, 4 evals each
    assert rec.fEvalsPerStep == pytest.approx(4.0)
    assert rec.solves == 0              # explicit: no internal solver


def testMultistepHistoryThreadingGivesNominalCost(case):
    """With `history` threaded (the suite's convention, via `result.history`)
    AB4 pays one Dormand-Prince bootstrap per warm-up step -- exactly
    `order - 1` of them -- and then 1 RHS evaluation per step. The starter's
    own cost is measured (not hard-coded) from a 1-step DP54 run."""
    ctx, system, dt = case
    spec = getSchemes(['ab4'], None)[0]
    dp1 = runScheme(ctx, system, getSchemes(['dp54'], None)[0], 1, dt,
                    warmup=0, keepFields=False).fEvals
    assert dp1 >= 5
    rec5 = runScheme(ctx, system, spec, 5, dt, warmup=0, keepFields=False)
    assert not rec5.diverged
    assert rec5.fEvals == 3 * dp1 + 2        # 3 bootstraps + 2 full-order steps
    # the marginal cost of a full-order AB4 step is one RHS evaluation
    rec7 = runScheme(ctx, system, spec, 7, dt, warmup=0, keepFields=False)
    assert rec7.fEvals - rec5.fEvals == 2
    # and the threaded average stays below the starter cost it amortizes
    assert rec7.fEvalsPerStep < dp1


def testImplicitSolverRecordedAndConverged(case):
    """The internal-solver record must be populated for an implicit run.

    Two invariants hold regardless of precision: the solve count is exactly
    stages x steps, and the RHS count exceeds the solve count (the difference
    is the GMRES matvec evaluations, which are RHS evaluations too). The
    iteration count is bounded by the solver's own budget + 1 (the final
    verification evaluation): under float32 the DIRK driver's weighted-norm
    convergence check is unreachable, so JFNK runs its full budget and
    `converged` stays False -- a property of the shipped stack, pinned here
    as an upper bound rather than an exact value so the test holds in
    float64 too, where JFNK converges in ~2 iterations.
    """
    ctx, system, dt = case
    rec = runScheme(ctx, system, getSchemes(['sdirk2_jfnk_jvp_1e-6'], None)[0],
                    3, dt, warmup=0, keepFields=False)
    assert not rec.diverged
    assert rec.solves == 2 * 3          # two DIRK stages x three steps
    assert 1 <= rec.itersMax <= 16      # max_iterations=15, +1 final evaluation
    assert rec.convergedSolves <= rec.solves
    assert rec.fEvals > rec.solves
    # Picard (fixed count, no tol) reports its schedule as "converged" by
    # construction -- the registry default's documented contract.
    recPic = runScheme(ctx, system, getSchemes(['sdirk2_picard2'], None)[0],
                       3, dt, warmup=0, keepFields=False)
    assert recPic.solves == 6
    assert recPic.convergedSolves == 6
    assert recPic.itersMax == 2


def testReuseBuiltSystemReproducible(case):
    """Two runs of the same scheme from the same built system must be
    bitwise identical -- the suite builds once and re-seeds, so a silent
    in-place mutation of the built state would corrupt every later run."""
    ctx, system, dt = case
    spec = getSchemes(['rk4'], None)[0]
    a = runScheme(ctx, system, spec, 2, dt, warmup=0)
    b = runScheme(ctx, system, spec, 2, dt, warmup=0)
    assert torch.equal(a.uFinal, b.uFinal)
    assert torch.equal(a.vFinal, b.vFinal)


def testAccuracyMachineryMeasuresConvergence(case):
    """The accuracy suite's comparison (rel-L2 against a finer reference)
    must actually shrink when the tested dt halves -- i.e. the machinery
    measures convergence rather than noise."""
    ctx, system, dt = case
    spec = getSchemes(['rk4'], None)[0]
    nRef = 8
    ref = runScheme(ctx, system, spec, nRef, dt / 4, warmup=0)
    coarse = runScheme(ctx, system, spec, nRef // 4, dt, warmup=0)
    fine = runScheme(ctx, system, spec, nRef // 2, dt / 2, warmup=0)
    errCoarse = relL2(coarse.uFinal, ref.uFinal)
    errFine = relL2(fine.uFinal, ref.uFinal)
    assert torch.isfinite(torch.tensor(errCoarse)) and errCoarse > 0
    assert errFine < errCoarse, f'expected dt/2 to beat dt: {errFine} !< {errCoarse}'


def testCaseParamOverridesReachTheCase():
    """`buildWaveCase(**overrides)` must route case parameters (not CaseSpec
    fields) onto `spec.params` the way `specFromArgs` does -- before the
    fix, `obstacleEnabled=False` raised `CaseSpec.__init__() got an
    unexpected keyword argument`. A wrong routing would silently keep the
    default (obstacle on) instead of disabling it."""
    ctxNoObs, systemNoObs, _ = buildWaveCase(nx=16, obstacleEnabled=False)
    ctxObs, systemObs, _ = buildWaveCase(nx=16, obstacleEnabled=True)
    assert bool(ctxNoObs.param('obstacleEnabled')) is False
    assert bool(ctxObs.param('obstacleEnabled')) is True
    # the obstacle changes the per-particle wave-speed grid, so the two
    # builds must actually differ (a no-op override would leave them equal)
    assert not torch.equal(systemNoObs.state.c, systemObs.state.c)


def testPerformanceRecordFields(case):
    """The fields the performance suite reports must be populated and
    self-consistent for a plain run."""
    ctx, system, dt = case
    rec = runScheme(ctx, system, getSchemes(['rk4'], None)[0], 2, dt,
                    warmup=1, keepFields=False)
    assert rec.msPerStep > 0.0
    assert rec.msPerRhs > 0.0
    assert rec.fEvalsPerStep == pytest.approx(4.0)
    assert rec.staticStateMB > 0.0
    assert rec.stepsDone == 3
    if torch.cuda.is_available() and str(ctx.device).startswith('cuda'):
        assert rec.peakAllocatedMB > 0.0
        assert rec.peakReservedMB >= rec.peakAllocatedMB
    # record serializes for the JSON output
    d = rec.toDict()
    assert 'uFinal' not in d and 'vFinal' not in d
    assert d['key'] == 'rk4'

