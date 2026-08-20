"""Unit tests for the implicit-shifting matrix-free solvers
(`modules/shifting/bicgstab.py::bicgstabSolve`,
`modules/shifting/gmres.py::gmresSolve`, and the opt-in last-resort
`modules/shifting/richardson.py::richardsonSolve`).

Dense CPU torch matrices, no warp/SPH involvement: the solvers take a
`matvec` closure and nothing else, and the shifting pipeline's own coverage
(`test_implicitShifting.py`, `test_implicitShiftingComparison.py`,
`test_implicitShiftingFallback.py`) sits on top of this. The case families
mirror what the production solve actually sees (see `ShiftingImplicitOperator`
and `docs/regression/implicit_shifting_operator_choice.md`): symmetric but
indefinite, singular with a translation-style null space, nonsymmetric
(non-uniform `omega_j`), and ill-conditioned (Jacobi-preconditioner path).

Shared status-code contract: `iters >= 0` = converged at that iterate;
`-10`/`-11` (BiCGStab only) rho/rv/omega breakdown; `-12` per-particle `|x|`
threshold bailout; `-13` (GMRES) stagnation / non-finite least squares;
`-14` max-iteration budget exhausted; `-15` (Richardson) stagnation.
`convergence[-1]` is always the verified true residual `||b - A x||` of the
returned iterate.

`richardsonSolve` is deliberately NOT in the Krylov `SOLVERS` list below: its
interface differs (no `precond` -- the production Jacobi diagonal diverges as
a Richardson step direction, see its module docstring -- and it adds
`omega`/`tune_omega`), so it is tested by its own dedicated cases.
"""

import pytest
import torch

from warpSPH.modules.shifting.bicgstab import bicgstabSolve
from warpSPH.modules.shifting.gmres import gmresSolve
from warpSPH.modules.shifting.richardson import richardsonSolve
from warpSPH.modules.shifting.preconditioner import buildScalarJacobiPrecond, buildBlockJacobiPrecond

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


# ---------------------------------------------------------------------------
# Preconditioner builders (modules/shifting/preconditioner.py) + the solvers'
# callable-precond path. The block builder is the dim-general [n, dim, dim]
# form; on the current operators it is a wash for scalar (see
# docs/regression/implicit_shifting_operator_choice.md), so these cases check
# correctness + robustness, not a convergence win.
# ---------------------------------------------------------------------------

def test_scalarJacobiMatchesInline():
    g = torch.Generator(device='cpu').manual_seed(0)
    diagBlock = torch.randn(10, 2, 2, generator=g)
    diagComponents = torch.diagonal(diagBlock, dim1=-2, dim2=-1).flatten()
    expected = torch.where(diagComponents.abs() > 1e-8, 1.0 / diagComponents, torch.zeros_like(diagComponents))
    assert torch.equal(buildScalarJacobiPrecond(diagBlock), expected)
    # all-~0 diagonal -> None (the historical `precond = None` branch)
    assert buildScalarJacobiPrecond(torch.zeros(5, 3, 3)) is None


def test_blockJacobiDim1ReducesToScalar():
    g = torch.Generator(device='cpu').manual_seed(1)
    d = torch.linspace(0.1, 3.0, 12)
    diagBlock = d.view(-1, 1, 1)
    r = torch.randn(12, generator=g)
    assert torch.allclose(buildBlockJacobiPrecond(diagBlock)(r), buildScalarJacobiPrecond(diagBlock) * r)


def test_blockJacobiAppliesBlockInverse():
    n, dim, seed = 6, 3, 2
    g = torch.Generator(device='cpu').manual_seed(seed)
    blocks = torch.stack([torch.randn(dim, dim, generator=g) for _ in range(n)])
    r = torch.randn(n * dim, generator=g).view(n, dim)
    got = buildBlockJacobiPrecond(blocks)(r.flatten()).view(n, dim)
    # the (tiny) ridge is negligible for these well-conditioned blocks
    exp = torch.stack([torch.linalg.inv(blocks[i]) @ r[i] for i in range(n)])
    assert torch.allclose(got, exp, atol=1e-4)


def test_blockJacobiNoneOnZeroBlocks():
    assert buildBlockJacobiPrecond(torch.zeros(4, 2, 2)) is None


def test_blockJacobiFiniteOnRankDeficientBlocks():
    # exactHessian-style: rank-deficient / indefinite blocks must invert
    # stably (via the scale-aware ridge) rather than producing Inf/NaN
    blocks = torch.stack([
        torch.tensor([[1.0, 1.0], [1.0, 1.0]]),   # rank 1 (null direction [1,-1])
        torch.tensor([[2.0, 0.0], [0.0, 0.0]]),   # rank 1
        torch.tensor([[-1.0, 0.5], [0.5, -0.3]]),  # indefinite
    ])
    out = buildBlockJacobiPrecond(blocks)(torch.randn(6))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize('solver, sid', zip(SOLVERS, SOLVER_IDS))
def test_solverCallablePrecondMatchesVector(solver, sid):
    # a callable `precond` must take over `psolve` and behave exactly like the
    # equivalent flat-vector `precond` (same preconditioned solve)
    n = 100
    g = torch.Generator(device='cpu').manual_seed(7)
    d = torch.linspace(1e-2, 1.0, n)
    A = torch.diag(d)
    b = A @ torch.randn(n, generator=g)
    vpre = 1.0 / d
    _xv, iv, _ = solver(_matvec(A), b, x0=torch.zeros(n), rtol=1e-6, maxiter=1000, precond=vpre)
    _xc, ic, _ = solver(_matvec(A), b, x0=torch.zeros(n), rtol=1e-6, maxiter=1000, precond=(lambda r: vpre * r))
    assert iv == ic
    assert torch.allclose(_xv, _xc)


@pytest.mark.parametrize('solver, sid', zip(SOLVERS, SOLVER_IDS))
def test_solverConvergesWithBlockPrecond(solver, sid):
    n, dim, seed = 40, 3, 8
    g = torch.Generator(device='cpu').manual_seed(seed)
    blocks = torch.stack([_randomSpd(dim, seed=seed + i) for i in range(n)])
    A = torch.zeros(n * dim, n * dim)
    for i in range(n):
        A[i * dim:(i + 1) * dim, i * dim:(i + 1) * dim] = blocks[i]
    b = A @ torch.randn(n * dim, generator=g)
    _x, it, _ = solver(_matvec(A), b, x0=torch.zeros(n * dim), rtol=1e-6, maxiter=1000,
                       precond=buildBlockJacobiPrecond(blocks))
    assert it >= 0
    assert _trueResid(A, b, _x) < 1e-4 * float(torch.linalg.norm(b))


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
    from warpSPH.configurations.moduleConfigurations.shifting import (
        ShiftProperties, ShiftingImplicitSolver, ShiftingImplicitFallback, ShiftingImplicitPreconditioner)
    p = ShiftProperties()
    assert p.implicitSolver is ShiftingImplicitSolver.bicgstab
    assert p.implicitTolerance == 0.0  # relative-only: the historical effective behavior
    assert p.implicitRestart == 30
    # the fallback chain is opt-in: the default must be `none` (legacy behavior)
    assert p.implicitFallback is ShiftingImplicitFallback.none
    # the preconditioner defaults to the historical scalar Jacobi, and the
    # null-space lift is off (0.0)
    assert p.implicitPreconditioner is ShiftingImplicitPreconditioner.scalar
    assert p.implicitNullSpaceLift == 0.0
    p.implicitFallback = ShiftingImplicitFallback.krylov
    assert p.implicitFallback.name == 'krylov'
    p.implicitSolver = ShiftingImplicitSolver.gmres
    assert p.implicitSolver.name == 'gmres'


# ---------------------------------------------------------------------------
# richardsonSolve (modules/shifting/richardson.py) -- the opt-in last-resort
# fallback. Dedicated cases (see the module docstring for why it is not in the
# Krylov SOLVERS list). The behavior mirrors an eigenvalue probe of the
# production operator: it converges on the positive-definite `legacyPairwise`
# regime, makes substantial (but slow) progress on an ill-conditioned
# singular system, and bails cleanly (finite, no NaN) on an indefinite one.
# ---------------------------------------------------------------------------

def test_richardsonConvergesSpd():
    n = 200
    A = _randomSpd(n)
    g = torch.Generator(device='cpu').manual_seed(20)
    b = A @ torch.randn(n, generator=g)
    x, iters, hist = richardsonSolve(_matvec(A), b, x0=torch.zeros(n), rtol=1e-6, maxiter=5000)
    assert iters >= 0, f'richardson did not converge (iters={iters})'
    assert _trueResid(A, b, x) < 1e-6 * torch.linalg.norm(b).item()
    # the final history entry is the verified true residual of the return
    assert hist[-1].item() == pytest.approx(_trueResid(A, b, x), rel=1e-6)


def test_richardsonConvergesNegDefinite():
    # A = -SPD: the step sign must be auto-detected as negative (backtracking
    # step search), and convergence still holds for the negative-definite case
    n = 100
    A = -_randomSpd(n)
    g = torch.Generator(device='cpu').manual_seed(21)
    b = A @ torch.randn(n, generator=g)
    x, iters, hist = richardsonSolve(_matvec(A), b, x0=torch.zeros(n), rtol=1e-6, maxiter=5000)
    assert iters >= 0, f'richardson did not converge on neg-def (iters={iters})'
    assert _trueResid(A, b, x) < 1e-6 * torch.linalg.norm(b).item()
    assert hist[-1].item() == pytest.approx(_trueResid(A, b, x), rel=1e-6)


def test_richardsonConvergesWellConditionedSingular():
    # the `legacyPairwise` regime: singular PSD with an exact translation null
    # space (b in range(A)). n=16 keeps the spectral gap wide enough that
    # Richardson finishes; the solution is determined up to the null space, so
    # the residual is the right thing to check
    n = 16
    A = _translationLaplacian(n)
    g = torch.Generator(device='cpu').manual_seed(22)
    xTrue = torch.randn(n, generator=g)
    xTrue = xTrue - xTrue.mean()
    b = A @ xTrue
    x, iters, _ = richardsonSolve(_matvec(A), b, x0=torch.zeros(n), rtol=1e-4, maxiter=5000)
    assert iters >= 0, f'richardson did not converge on singular (iters={iters})'
    assert _trueResid(A, b, x) < 1e-4 * torch.linalg.norm(b).item()


def test_richardsonMakesProgressOnIllConditioned():
    # an n=100 Laplacian is far too ill-conditioned for Richardson to finish in
    # a moderate budget -- the documented last-resort behavior: substantial
    # residual reduction, a negative status, and a finite iterate (never a NaN
    # or a hang)
    n = 100
    A = _translationLaplacian(n)
    g = torch.Generator(device='cpu').manual_seed(23)
    xTrue = torch.randn(n, generator=g)
    xTrue = xTrue - xTrue.mean()
    b = A @ xTrue
    bnorm = torch.linalg.norm(b).item()
    x, iters, hist = richardsonSolve(_matvec(A), b, x0=torch.zeros(n), rtol=1e-6, maxiter=200)
    assert iters < 0
    assert torch.isfinite(x).all()
    assert hist[-1].item() < 0.1 * bnorm, 'must make substantial progress even when it cannot finish'


def test_richardsonIndefiniteBailsFinite():
    # symmetric indefinite: no Richardson step size converges -> a negative
    # status (stagnation), a finite iterate, and a valid stamped residual
    n = 100
    A = _symmetricIndefinite(n)
    g = torch.Generator(device='cpu').manual_seed(24)
    b = A @ torch.randn(n, generator=g)
    x, iters, hist = richardsonSolve(_matvec(A), b, x0=torch.zeros(n), rtol=1e-8, maxiter=200)
    assert iters < 0
    assert torch.isfinite(x).all()
    assert hist[-1].item() == pytest.approx(_trueResid(A, b, x), rel=1e-6)


def test_richardsonStatusBudgetExhausted():
    # rtol below fp32 machine precision: unreachable, so the solve must burn
    # its budget and report -14 (not a converged-style status)
    n = 100
    A = _randomSpd(n)
    b = A @ torch.randn(n)
    x, iters, hist = richardsonSolve(_matvec(A), b, x0=torch.zeros(n), rtol=1e-9, maxiter=5)
    assert iters == -14
    assert hist[-1].item() == pytest.approx(_trueResid(A, b, x), rel=1e-6)


def test_richardsonStatusThresholdBailout():
    # 2x2 whose exact solution (10, -5) violates threshold=5 (|10| > 5): the
    # solve must bail with -12 (it cannot converge while staying under the
    # threshold) and stamp the true residual
    A = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    b = torch.tensor([10.0, 0.0])
    x, iters, hist = richardsonSolve(_matvec(A), b, x0=torch.zeros(2), tol=0.0, rtol=0.0,
                                     maxiter=200, threshold=5.0, dim=1,
                                     tune_omega=False, omega=0.5)
    assert iters == -12
    assert hist[-1].item() == pytest.approx(_trueResid(A, b, x), rel=1e-5)


def test_richardsonZeroRhs():
    A = _randomSpd(32)
    b = torch.zeros(32)
    x, iters, hist = richardsonSolve(_matvec(A), b)
    assert iters == 0
    assert torch.equal(x, b)
    assert hist == []


def test_richardsonExplicitOmegaSkipsTuning():
    # passing omega explicitly (tune_omega=False) uses it verbatim; 0.5/rho(A)
    # is inside the Richardson window for this SPD system, so it converges
    n = 100
    A = _randomSpd(n)
    g = torch.Generator(device='cpu').manual_seed(25)
    b = A @ torch.randn(n, generator=g)
    omega = 0.5 / torch.linalg.norm(A, ord=2).item()
    x, iters, _ = richardsonSolve(_matvec(A), b, x0=torch.zeros(n), rtol=1e-6, maxiter=5000,
                                  tune_omega=False, omega=omega)
    assert iters >= 0
    assert _trueResid(A, b, x) < 1e-6 * torch.linalg.norm(b).item()


def test_richardsonWarmStartNoSlowerThanCold():
    # the outer solveShifting loop re-solves a nearly-identical system every
    # iteration: a warm start (the previous solve's answer) must never be worse
    # than a cold start
    n = 100
    A = _randomSpd(n)
    g = torch.Generator(device='cpu').manual_seed(26)
    b = A @ torch.randn(n, generator=g)
    xWarm = torch.linalg.solve(A, 1.01 * b)  # the "previous iteration's" answer
    _x, iCold, _ = richardsonSolve(_matvec(A), b, x0=torch.zeros(n), rtol=1e-6, maxiter=5000)
    x, iWarm, _ = richardsonSolve(_matvec(A), b, x0=xWarm, rtol=1e-6, maxiter=5000)
    assert iCold >= 0 and iWarm >= 0
    assert iWarm <= iCold, f'warm start ({iWarm}) slower than cold ({iCold})'
    assert _trueResid(A, b, x) < 1e-6 * torch.linalg.norm(b).item()


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))