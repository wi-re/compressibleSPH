"""Unit tests for the implicit-shifting matrix-free Krylov solvers
(`modules/shifting/bicgstab.py::bicgstabSolve` and
`modules/shifting/gmres.py::gmresSolve`).

Dense CPU torch matrices, no warp/SPH involvement: the solvers take a
`matvec` closure and nothing else, and the shifting pipeline's own coverage
(`test_implicitShifting.py`, `test_implicitShiftingComparison.py`) sits on
top of this. The case families mirror what the production solve actually
sees (see `ShiftingImplicitOperator` and
`docs/regression/implicit_shifting_operator_choice.md`): symmetric but
indefinite, singular with a translation-style null space, nonsymmetric
(non-uniform `omega_j`), and ill-conditioned (Jacobi-preconditioner path).

Shared status-code contract: `iters >= 0` = converged at that iterate;
`-10`/`-11` (BiCGStab only) rho/rv/omega breakdown; `-12` per-particle
`|x|` threshold bailout; `-13` (GMRES) stagnation / non-finite least
squares; `-14` max-iteration budget exhausted. `convergence[-1]` is always
the verified true residual `||b - A x||` of the returned iterate.
"""

import pytest
import torch

from warpSPH.modules.shifting.bicgstab import bicgstabSolve
from warpSPH.modules.shifting.gmres import gmresSolve

SOLVERS = [bicgstabSolve, gmresSolve]
SOLVER_IDS = ['bicgstab', 'gmres']


def _matvec(A):
    return lambda x: A @ x


def _trueResid(A, b, x):
    return torch.linalg.norm(A @ x - b).item()


def _randomSpd(n, seed=0, dtype=torch.float32):
    g = torch.Generator(device='cpu').manual_seed(seed)
    M = torch.randn(n, n, generator=g)
    return (M @ M.T + n * torch.eye(n)).to(dtype)


def _randomNonsymmetric(n, seed=0, dtype=torch.float32):
    g = torch.Generator(device='cpu').manual_seed(seed)
    M = torch.randn(n, n, generator=g)
    # well-conditioned: I + scaled random keeps the spectrum off the origin
    return (torch.eye(n) + 0.5 * M / torch.linalg.norm(M, ord=2)).to(dtype)


def _symmetricIndefinite(n, seed=0, dtype=torch.float32):
    # well-conditioned symmetric indefinite: eigenvalues exactly +-1 (the
    # spectrum straddles zero with a wide gap, so rtol=1e-6 is reachable in
    # fp32; a median-centered random spectrum leaves an O(1/n) gap near zero
    # that no fp32 Krylov solver can get past)
    g = torch.Generator(device='cpu').manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    lam = torch.ones(n)
    lam[: n // 2] = -1.0
    return (Q @ torch.diag(lam) @ Q.T).to(dtype)


def _translationLaplacian(n, dtype=torch.float32):
    # path-graph Laplacian: symmetric, PSD, exact null space span(1)
    i = torch.arange(n - 1)
    A = torch.zeros(n, n)
    A[i, i] += 1
    A[i + 1, i + 1] += 1
    A[i, i + 1] -= 1
    A[i + 1, i] -= 1
    return A.to(dtype)


@pytest.mark.parametrize('solver, sid', zip(SOLVERS, SOLVER_IDS))
def test_converges_spd(solver, sid):
    n = 200
    A = _randomSpd(n)
    g = torch.Generator(device='cpu').manual_seed(1)
    xTrue = torch.randn(n, generator=g)
    b = A @ xTrue
    x, iters, hist = solver(_matvec(A), b, x0=torch.zeros(n), rtol=1e-6, maxiter=2000)
    assert iters >= 0, f'{sid}: did not converge (iters={iters})'
    assert _trueResid(A, b, x) < 1e-6 * torch.linalg.norm(b).item()
    # the final history entry is the verified true residual of the return
    assert hist[-1].item() == pytest.approx(_trueResid(A, b, x), rel=1e-6)


@pytest.mark.parametrize('solver, sid', zip(SOLVERS, SOLVER_IDS))
def test_converges_nonsymmetric(solver, sid):
    n = 150
    A = _randomNonsymmetric(n)
    g = torch.Generator(device='cpu').manual_seed(2)
    xTrue = torch.randn(n, generator=g)
    b = A @ xTrue
    x, iters, hist = solver(_matvec(A), b, x0=torch.zeros(n), rtol=1e-6, maxiter=2000)
    assert iters >= 0, f'{sid}: did not converge (iters={iters})'
    assert _trueResid(A, b, x) < 1e-6 * torch.linalg.norm(b).item()


@pytest.mark.parametrize('solver, sid', zip(SOLVERS, SOLVER_IDS))
def test_converges_symmetricIndefinite(solver, sid):
    # rtol=2e-6, not 1e-6: this system's fp32 stagnation floor is ~1.1e-6
    # (the same solve reaches 7e-7 in one iterate in fp64); the old port
    # used to declare a false recurrence-based "convergence" at 1.8e-6 here.
    n = 100
    A = _symmetricIndefinite(n)
    g = torch.Generator(device='cpu').manual_seed(3)
    xTrue = torch.randn(n, generator=g)
    b = A @ xTrue
    x, iters, _hist = solver(_matvec(A), b, x0=torch.zeros(n), rtol=2e-6, maxiter=4000)
    assert iters >= 0, f'{sid}: did not converge (iters={iters})'
    assert _trueResid(A, b, x) < 2e-6 * torch.linalg.norm(b).item()


@pytest.mark.parametrize('solver, sid', zip(SOLVERS, SOLVER_IDS))
def test_converges_singularTranslationNullspace(solver, sid):
    # the `exactHessian` regime: consistent singular system (b in range(A),
    # exact translation null space) -- the residual is the right thing to
    # check; the solution is only determined up to the null space
    n = 200
    A = _translationLaplacian(n)
    g = torch.Generator(device='cpu').manual_seed(4)
    xTrue = torch.randn(n, generator=g)
    xTrue = xTrue - xTrue.mean()
    b = A @ xTrue
    x, iters, _hist = solver(_matvec(A), b, x0=torch.zeros(n), rtol=1e-5, maxiter=5000)
    assert iters >= 0, f'{sid}: did not converge (iters={iters})'
    assert _trueResid(A, b, x) < 1e-5 * torch.linalg.norm(b).item()


@pytest.mark.parametrize('solver, sid', zip(SOLVERS, SOLVER_IDS))
def test_warmStartNoSlowerThanCold(solver, sid):
    # the outer solveShifting loop re-solves a nearly-identical system every
    # iteration: x0 = the previous (slightly different) solve's answer must
    # never be worse than a cold start. (On a random SPD system GMRES's
    # per-cycle reduction is large enough that both fit in one m-cycle, so
    # the honest assertion is "<= plus both converge", not a strict speedup.)
    n = 200
    A = _randomSpd(n)
    g = torch.Generator(device='cpu').manual_seed(5)
    b = A @ torch.randn(n, generator=g)
    xWarm = torch.linalg.solve(A, 1.01 * b)  # the "previous iteration's" answer
    _x, iCold, _ = solver(_matvec(A), b, x0=torch.zeros(n), rtol=1e-6, maxiter=3000)
    x, iWarm, _ = solver(_matvec(A), b, x0=xWarm, rtol=1e-6, maxiter=3000)
    assert iCold >= 0 and iWarm >= 0
    assert iWarm <= iCold, f'{sid}: warm start ({iWarm}) slower than cold ({iCold})'
    assert _trueResid(A, b, x) < 1e-6 * torch.linalg.norm(b).item()


@pytest.mark.parametrize('solver, sid', zip(SOLVERS, SOLVER_IDS))
def test_jacobiPreconditionerSpeedsUp(solver, sid):
    n = 150
    g = torch.Generator(device='cpu').manual_seed(6)
    d = torch.linspace(1e-3, 1.0, n)
    M = torch.randn(n, n, generator=g)
    A = torch.diag(d) + 0.01 * (M @ M.T) / n  # SPD, diagonal-dominated
    xTrue = torch.randn(n, generator=g)
    b = A @ xTrue
    precond = 1.0 / torch.diagonal(A)
    _x, iPre, _ = solver(_matvec(A), b, x0=torch.zeros(n), rtol=1e-6, maxiter=1000, precond=precond)
    _x, iNone, _ = solver(_matvec(A), b, x0=torch.zeros(n), rtol=1e-6, maxiter=1000)
    assert iPre >= 0
    assert iPre < iNone or iNone < 0, f'{sid}: preconditioner not faster ({iPre} vs {iNone})'


@pytest.mark.parametrize('solver, sid', zip(SOLVERS, SOLVER_IDS))
def test_statusZeroRhs(solver, sid):
    A = _randomSpd(32)
    b = torch.zeros(32)
    x, iters, hist = solver(_matvec(A), b)
    assert iters == 0
    assert torch.equal(x, b)
    assert hist == []


@pytest.mark.parametrize('solver, sid', zip(SOLVERS, SOLVER_IDS))
def test_statusBudgetExhausted(solver, sid):
    # rtol=1e-9 is below fp32 machine precision: unreachable, so the solve
    # must burn its budget and report -14 (not a converged-style status)
    A = _randomSpd(100)
    b = A @ torch.randn(100)
    x, iters, hist = solver(_matvec(A), b, x0=torch.zeros(100), rtol=1e-9, maxiter=5)
    assert iters == -14
    # the returned iterate is still stamped with its true residual
    assert hist[-1].item() == pytest.approx(_trueResid(A, b, x), rel=1e-6)


def test_bicgstabStatusThresholdBailout():
    # 2x2 system whose exact solution (10, -5) violates threshold=5: the
    # solve must bail with -12 (not converge) and report the true residual
    A = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    b = torch.tensor([10.0, 0.0])
    x, iters, hist = bicgstabSolve(_matvec(A), b, x0=torch.zeros(2), tol=0.0, rtol=0.0,
                                   maxiter=100, threshold=5.0, dim=1)
    assert iters == -12
    assert torch.allclose(x, torch.tensor([10.0, -4.0]), atol=1e-5)
    assert hist[-1].item() == pytest.approx(5.0 ** 0.5, rel=1e-5)


def test_gmresStatusThresholdBailout():
    # same 2x2 system: with n = 2, GMRES runs m = min(restart, n-1) = 1 and
    # its first cycle-boundary LS update is (8, 0), which also violates
    # threshold=5
    A = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    b = torch.tensor([10.0, 0.0])
    x, iters, hist = gmresSolve(_matvec(A), b, x0=torch.zeros(2), tol=0.0, rtol=0.0,
                                maxiter=100, threshold=5.0, dim=1)
    assert iters == -12
    assert torch.allclose(x, torch.tensor([8.0, 0.0]), atol=1e-5)
    assert hist[-1].item() == pytest.approx(20.0 ** 0.5, rel=1e-5)


def test_bicgstabSConvergenceReportsIteration():
    # regression: the port's s-convergence branch returned a literal 0 at
    # every iterate; a solve that converges via |s| < atol mid-run must
    # report the actual iterate count
    n = 60
    A = torch.diag(torch.linspace(1.0, 10.0, n)).double()
    g = torch.Generator(device='cpu').manual_seed(7)
    b = torch.randn(n, generator=g, dtype=torch.float64)
    x, iters, _hist = bicgstabSolve(lambda xx: A @ xx, b, x0=torch.zeros(n, dtype=torch.float64),
                                    tol=1e-12, rtol=0.0, maxiter=200)
    assert 1 <= iters < 200, f's-convergence must report its iterate, got {iters}'
    assert _trueResid(A, b, x) < 1e-12


def test_bicgstabHonorsAbsoluteTol():
    # regression: `tol` used to be accepted but never read, so rtol=0 meant
    # atol=0 and the solve could never converge; the absolute floor must now
    # actually terminate the solve. b is scaled to unit norm so the absolute
    # target sits comfortably above the fp32 floor.
    n = 100
    g = torch.Generator(device='cpu').manual_seed(8)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    lam = torch.ones(n)
    lam[::2] = 0.9
    A = Q @ torch.diag(lam) @ Q.T
    b = torch.randn(n, generator=g)
    b = b / torch.linalg.norm(b)
    x, iWithTol, _ = bicgstabSolve(_matvec(A), b, x0=torch.zeros(n), tol=1e-6, rtol=0.0, maxiter=500)
    assert iWithTol >= 0, f'tol=1e-6 must terminate the solve, got {iWithTol}'
    _x, iNoTol, _ = bicgstabSolve(_matvec(A), b, x0=torch.zeros(n), tol=0.0, rtol=0.0, maxiter=500)
    assert iNoTol < 0, 'atol == 0: no convergence test can ever pass'


def test_gmresRestartConverges():
    # the system needs more than one m-cycle: convergence must survive the
    # cycle boundary (LS update + true-residual restart)
    A = _randomSpd(300, seed=9)
    g = torch.Generator(device='cpu').manual_seed(10)
    b = A @ torch.randn(300, generator=g)
    x, iters, hist = gmresSolve(_matvec(A), b, x0=torch.zeros(300), rtol=1e-6, maxiter=500, restart=10)
    assert iters >= 0, f'GMRES(10) did not converge (iters={iters})'
    assert iters > 10, 'expected at least one full restart cycle'
    assert _trueResid(A, b, x) < 1e-6 * torch.linalg.norm(b).item()
    assert hist[-1].item() == pytest.approx(_trueResid(A, b, x), rel=1e-6)


def test_gmresInconsistentSystemBails():
    # nilpotent A with b off range(A): GMRES(1) reaches the (1-dim)
    # invariant subspace without converging -- the least-squares block is
    # singular, so the solve must bail with -13 (not NaN, not hang)
    A = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
    b = torch.tensor([1.0, 1.0], dtype=torch.float64)
    x, iters, hist = gmresSolve(lambda xx: A @ xx, b, x0=torch.zeros(2, dtype=torch.float64),
                                tol=1e-12, rtol=0.0, maxiter=100, restart=1)
    assert iters == -13
    assert torch.isfinite(x).all()


def test_solversAgreeOnSpd():
    n = 100
    A = _randomSpd(n, dtype=torch.float64)
    g = torch.Generator(device='cpu').manual_seed(11)
    b = A @ torch.randn(n, generator=g, dtype=torch.float64)
    xB, iB, _ = bicgstabSolve(_matvec(A), b, x0=torch.zeros(n, dtype=torch.float64), rtol=1e-10, maxiter=5000)
    xG, iG, _ = gmresSolve(_matvec(A), b, x0=torch.zeros(n, dtype=torch.float64), rtol=1e-10, maxiter=5000)
    assert iB >= 0 and iG >= 0
    assert torch.linalg.norm(xB - xG) < 1e-8 * torch.linalg.norm(xB)


def test_configPlumbingDefaults():
    from warpSPH.configurations.moduleConfigurations.shifting import ShiftProperties, ShiftingImplicitSolver
    p = ShiftProperties()
    assert p.implicitSolver is ShiftingImplicitSolver.bicgstab
    assert p.implicitTolerance == 0.0  # relative-only: the historical effective behavior
    assert p.implicitRestart == 30
    p.implicitSolver = ShiftingImplicitSolver.gmres
    assert p.implicitSolver.name == 'gmres'


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))